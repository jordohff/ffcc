"""Blend this project's own draft board with trusted external ranking
sources into a composite/consensus comparison board - built 2026-08-31 at
the user's request, using files they hand-downloaded into
data/paid/ConsensusRankings/ (gitignored, same treatment as every other
paid source in this project - see paid_data.py).

Sources, as described by the user:
- Dataroma: overall + per-position rank, PPR, has real ADP for comparison.
  "Has a little more juice on it" (the user's most-trusted source here).
- Scott Barrett (FantasyPoints): overall rank only, scoring format NOT
  stated in the export - ASSUMED PPR (same site/subscription as the Hansen
  file below, which IS explicitly PPR) but not confirmed. Flagged, not
  silently trusted.
- John Hansen (FantasyPoints): overall rank (top 200), explicitly PPR (a
  "Format" column in the export, verified uniformly "ppr" across all 200
  rows).
- The Coachspeak Index / Greg Brainos: POSITIONAL ONLY (no overall rank),
  half-PPR, tiered (a blank line between rank groups = a real tier break,
  per the user). Delivered as a PDF, not a CSV - parsed via pdftotext.
- Joel Smyth and Hayden Winks (added 2026-09-01): overall + derived position
  rank, both half-PPR and full-PPR exports available (like Dataroma).
  Delivered as .docx files (a simple 4-column Rank/Player/Position/Team
  Word table) - parsed via the same zipfile+XML approach as coachspeak.py,
  not a new dependency. Same weight as Dataroma/Barrett/Hansen (see
  SOURCE_WEIGHTS), per the user's explicit instruction.
- Jeff Bell and Sigmund Bloom (added 2026-09-02): overall + position rank,
  PPR only (no half-PPR export provided - reused for both scoring formats,
  same fallback precedent as Dataroma). Delivered as .xlsx, an identical
  format between the two - a "Rank" column that's either a real overall-
  rank integer or a "Tier N" label row (no player of its own - a tier
  break, not a rank), a "Player" column combining name+team ("Ja'Marr
  Chase CIN"), and a "Pos" column combining position+position-rank
  ("WR1", "PK28", "TD32" - this source's own "PK"/"TD" labels, remapped
  to "K"/"DST" here). Both cover the FULL player pool including K/DST
  (32 kickers, 32 team defenses each) - the primary source for this
  project's new K/DST consensus (see build_kdst_consensus_board).
- Josh Norris (added 2026-09-02): same 4-column docx format as Smyth/
  Winks (via load_docx_rankings), both half-PPR and full-PPR exports.
  Ranks ~300 overall, which happens to include exactly 1 K and 1 DST at
  the tail - a negligible but real K/DST contribution on top of Bell/
  Bloom/Winks.

K/DST: this project's own model doesn't project kickers or defenses at
all (out of scope from the start - see CLAUDE.md's Data Sources section),
so there's no board to anchor a K/DST composite to the way
build_composite_board anchors QB/RB/WR/TE to our own model's player set.
Built as a SEPARATE function, build_kdst_consensus_board, from whichever
sources actually rank K/DST (Bell/Bloom full coverage, Winks partial,
Norris negligible - Dataroma/Barrett/Hansen/CSI don't cover K/DST at
all, confirmed by inspecting each export's own position list) - a union
of every K/DST player ANY of those sources ranks, not anchored to one
source's list.

Design choices, and why:
- WEIGHTED across sources (see SOURCE_WEIGHTS) - started equal-weight
  (2026-08-31), revised same day to half-weight our model + CSI, revised
  again the same day to quarter-weight our model specifically (still walk-
  forward validating, and the user has low confidence in its durability
  estimates for players with a fresh real-world opportunity change - see
  SOURCE_WEIGHTS' own comment for the full reasoning), all per the user's
  explicit, evolving choice.
- Anchored to OUR OWN board's player set (the already-built 740-row
  draft_rankings CSV), not a new universe - an external source's player we
  don't already have a row for is dropped, not added. This project's board
  is already very deep (every fantasy-relevant player and most of the
  replacement-level pool), so this is a small, known scope limitation, not
  a real gap in practice.
- TWO separate composite numbers, not one blended score - mixing them would
  hide real information:
    - consensus_position_pct: average of each source's own WITHIN-POSITION
      percentile (position_rank / that source's own position pool size).
      Computable for every source, including CSI (positional-only) - this
      is the only number CSI feeds into.
    - consensus_overall_rank: average of each source's raw overall rank,
      only across sources that actually publish one (our model + Dataroma +
      Barrett + Hansen). Familiar "average rank" / ECR-style number. CSI is
      deliberately EXCLUDED here rather than inventing an implied overall
      rank for a positional-only source - see the module's own discussion
      in build_composite_board.
"""

import difflib
import re
import subprocess
import unicodedata
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd

from .data import TEAM_CODE_FIXES
from .paid_data import _normalize_name

# Jacksonville is coded "JAC" by some of these sources (Bell/Bloom, Winks)
# and "JAX" by others (Norris) - not covered by data.TEAM_CODE_FIXES (that
# map normalizes nflreadpy's OWN team-code inconsistencies, a different,
# unrelated set of sources). Extended locally rather than touching the
# shared project-wide map for a quirk specific to these consensus exports.
_KDST_TEAM_FIXES = {**TEAM_CODE_FIXES, "JAC": "JAX"}


def _strip_accents(s: str) -> str:
    """"Piñeiro" -> "Pineiro" - some sources drop/mangle the accent
    (confirmed: Bell/Bloom's export shows it as a mojibake "�", Winks/Norris
    show it as a clean "Pineiro") while others keep it, which otherwise
    scatters the same real kicker across 2 unmatched K/DST consensus rows.
    Not applied to _normalize_name globally (used throughout this project
    for QB/RB/WR/TE matching, where this hasn't been observed as a problem)
    - scoped to the K/DST consensus path where it was actually found.
    """
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

POSITION_HEADERS = {"RB", "WR", "TE", "QB"}

_JUNK_LINE_RE = re.compile(r"^(ff|fi|fl)(\s+(ff|fi|fl))*$", re.IGNORECASE)
_RANK_ENTRY_RE = re.compile(r"(\d+)\.\s*([A-Za-z][A-Za-z.'\- ]*?)(?=\s*\d+\.|$)")


def _read_csi_text(pdf_path: str) -> str:
    """Extract raw (NOT -layout) text from the CSI PDF via pdftotext. The
    source is a 3-column-per-position grid; -layout mode tries to preserve
    x-position and scrambles reading order into a row-major mess across the
    3 columns, while raw extraction follows the PDF content stream, which
    for this file happens to match the rank-number sequence directly
    (verified against the full 08/30/26 export before trusting it) - so raw
    mode is used deliberately, not because -layout was untried.
    """
    result = subprocess.run(["pdftotext", pdf_path, "-"], capture_output=True, text=True, check=True)
    return result.stdout


def parse_csi_rankings(pdf_path: str) -> pd.DataFrame:
    """Parse The Coachspeak Index's tiered positional PDF into one row per
    player: [raw_name, position, csi_position_rank, csi_tier]. Positional
    only - no overall rank exists in this source.

    Real quirks handled here (verified against the 08/30/26 export, not
    assumed):
    - A blank line between rank groups is a real tier break (per the user),
      tracked as csi_tier - incremented once per blank line within a
      position section; consecutive blanks don't double-count, and a
      position-header change resets the counter rather than adding a
      spurious extra tier at the section boundary.
    - Standalone lines containing only stray "ff"/"fi"/"fl" ligature
      fragments are a page-footer/logo artifact (not tier signal) and are
      dropped before tier-break detection - confirmed these never carry
      real rank content (verified by inspecting every occurrence in the
      source file).
    - The same ligature glyphs are ALSO dropped MID-NAME for a few players
      (e.g. "Christian McCa rey" for "McCaffrey", "Justin Je erson" for
      "Jefferson") - NOT reconstructed here, since guessing where a
      ligature was dropped is fragile; left for match_to_board's fuzzy
      fallback, which can recover these via closeness to a real board name
      restricted to the same position.
    - One name wraps across a line break with a hyphen ("KeAndre Lambert-"
      / "Smith") - a line ending in "-" is joined directly (no space) to
      the next line before rank-token parsing.
    """
    text = _read_csi_text(pdf_path)
    rows: list[dict] = []
    position = None
    tier = 0
    pending_tier_break = False
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        blob = ""
        for line in buffer:
            if blob.endswith("-"):
                blob += line
            elif blob:
                blob += " " + line
            else:
                blob = line
        for m in _RANK_ENTRY_RE.finditer(blob):
            name = m.group(2).strip().rstrip(".")
            if name:
                rows.append({"raw_name": name, "position": position, "csi_position_rank": int(m.group(1)), "csi_tier": tier})
        buffer.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            if position is not None:
                pending_tier_break = True
            continue
        if line in POSITION_HEADERS:
            flush()
            position = line
            tier = 0
            pending_tier_break = False
            continue
        if position is None or _JUNK_LINE_RE.match(line) or "Coachspeak" in line or "REDRAFT" in line:
            continue
        if pending_tier_break:
            tier += 1
            pending_tier_break = False
        buffer.append(line)

    flush()
    return pd.DataFrame(rows)


def load_csi(pdf_path: str) -> pd.DataFrame:
    """CSI/Greg Brainos rankings, ready for matching: adds normalized_name.
    Half-PPR, positional only (no overall rank) - see module docstring.
    """
    df = parse_csi_rankings(pdf_path)
    df["normalized_name"] = df["raw_name"].map(_normalize_name)
    return df


def _extract_docx_table_cells(docx_path: str) -> list[str]:
    """Flat, in-document-order list of paragraph text runs from a .docx -
    reused (not shared/imported) from the same zipfile+XML approach already
    used for coachspeak.py's Discord export, since that helper is private to
    a module with a different job. Works for a simple Word TABLE export the
    same way it works for plain paragraphs, since `w:p` appears inside table
    cells too - each cell becomes one entry in document order.
    """
    with zipfile.ZipFile(docx_path) as z:
        xml_bytes = z.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    cells = []
    for p in root.iter(f"{_W_NS}p"):
        texts = [t.text or "" for t in p.iter(f"{_W_NS}t")]
        cells.append("".join(texts))
    return cells


def load_docx_rankings(docx_path: str, prefix: str) -> pd.DataFrame:
    """Parse a simple 4-column Word-table ranking export (Rank / Player /
    Position / Team, one player per row) into a DataFrame with
    raw_name/position/team/{prefix}_overall_rank/{prefix}_position_rank.

    Used for both Joel Smyth's and Hayden Winks's ranking exports (added
    2026-09-01) - confirmed both share this exact structure (a leading
    blank paragraph, a 4-cell header, then repeating rank/player/position/
    team groups, a trailing blank) before writing one shared loader instead
    of two near-duplicate ones. Position rank is DERIVED (not published),
    same convention as load_barrett/load_hansen.
    """
    cells = _extract_docx_table_cells(docx_path)
    # Drop leading/trailing blanks and the 4-cell header row.
    cells = [c for c in cells if c != ""]
    header, body = cells[:4], cells[4:]
    if header != ["Rank", "Player", "Position", "Team"]:
        raise ValueError(f"{docx_path}: unexpected header {header!r} - format may have changed")
    if len(body) % 4 != 0:
        raise ValueError(f"{docx_path}: body length {len(body)} isn't a multiple of 4 - format may have changed")
    rows = [body[i:i + 4] for i in range(0, len(body), 4)]
    df = pd.DataFrame(rows, columns=["overall_rank", "raw_name", "position", "team"])
    df = df.rename(columns={"overall_rank": f"{prefix}_overall_rank"})
    df[f"{prefix}_overall_rank"] = df[f"{prefix}_overall_rank"].astype(int)
    df[f"{prefix}_position_rank"] = df.groupby("position")[f"{prefix}_overall_rank"].rank(method="first").astype(int)
    df["normalized_name"] = df["raw_name"].map(_normalize_name)
    return df


def load_smyth(path: str) -> pd.DataFrame:
    """Joel Smyth's redraft rankings (added 2026-09-01, same weight as
    Dataroma/Barrett/Hansen - see SOURCE_WEIGHTS)."""
    return load_docx_rankings(path, "smyth")


def load_winks(path: str) -> pd.DataFrame:
    """Hayden Winks's redraft rankings (added 2026-09-01, same weight as
    Dataroma/Barrett/Hansen - see SOURCE_WEIGHTS)."""
    return load_docx_rankings(path, "winks")


_TIERED_POS_RE = re.compile(r"^([A-Za-z]+?)(\d+)$")
_KDST_POSITION_MAP = {"PK": "K", "TD": "DST"}


def _split_trailing_team(raw_player: str) -> tuple[str, str]:
    """"Ja'Marr Chase CIN" -> ("Ja'Marr Chase", "CIN") - team is always the
    last whitespace-separated token in the Bell/Bloom export (verified: even
    a name with a real suffix, e.g. "Stetson Bennett IV LAR", keeps the
    suffix as part of the name since "IV" isn't a team code - the true team
    code is still the very last token).
    """
    parts = raw_player.rsplit(" ", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return raw_player.strip(), ""


def load_tiered_xlsx_rankings(path: str, prefix: str) -> pd.DataFrame:
    """Parse Jeff Bell's / Sigmund Bloom's ranking export (added 2026-09-02,
    identical structure between the two, verified before writing one shared
    loader): column 0 ("Rank") is EITHER a real overall-rank integer or a
    "Tier N" label row (no player/position of its own - a tier break, not a
    player) - deliberately read positionally (raw.iloc[:, 0]), not via
    raw["Rank"], since the file ALSO has a second, differently-purposed
    "Rank\\nvs ADP" column that collides on the name "Rank" once a header's
    embedded newline is stripped. "Player" combines name+team
    ("Ja'Marr Chase CIN" - see _split_trailing_team); "Pos" combines
    position+position-rank ("WR1", "PK28", "TD32" - this source's own
    "PK"/"TD" kicker/defense labels, remapped to this project's "K"/"DST").
    Real overall rank runs through the WHOLE player pool, K/DST included
    (confirmed: e.g. rank 444-458 covers the export's last several K/DST/
    backup-QB rows) - usable directly as an overall_rank, same as
    Dataroma's.
    """
    raw = pd.read_excel(path)
    rows = []
    tier = 0
    for _, row in raw.iterrows():
        rank_val = row.iloc[0]
        player, pos = row["Player"], row["Pos"]
        if pd.isna(player) or pd.isna(pos):
            if isinstance(rank_val, str) and rank_val.strip().lower().startswith("tier"):
                tier += 1
            continue
        name, team = _split_trailing_team(str(player))
        m = _TIERED_POS_RE.match(str(pos).strip())
        if not m:
            continue
        pos_label = _KDST_POSITION_MAP.get(m.group(1), m.group(1))
        rows.append({
            "raw_name": name, "team": team, "position": pos_label,
            f"{prefix}_overall_rank": int(rank_val), f"{prefix}_position_rank": int(m.group(2)),
            f"{prefix}_tier": tier,
        })
    df = pd.DataFrame(rows)
    df["normalized_name"] = df["raw_name"].map(_normalize_name)
    return df


def load_bell(path: str) -> pd.DataFrame:
    """Jeff Bell's redraft rankings (added 2026-09-02, PPR only - reused for
    both scoring formats, same fallback precedent as Dataroma). Same weight
    as Dataroma/Barrett/Hansen/Smyth/Winks (see SOURCE_WEIGHTS)."""
    return load_tiered_xlsx_rankings(path, "bell")


def load_bloom(path: str) -> pd.DataFrame:
    """Sigmund Bloom's redraft rankings (added 2026-09-02, PPR only - reused
    for both scoring formats, same fallback precedent as Dataroma). Same
    weight as Dataroma/Barrett/Hansen/Smyth/Winks (see SOURCE_WEIGHTS)."""
    return load_tiered_xlsx_rankings(path, "bloom")


def load_norris(path: str) -> pd.DataFrame:
    """Josh Norris's redraft rankings (added 2026-09-02) - same 4-column
    Rank/Player/Position/Team docx format as Smyth/Winks, reuses
    load_docx_rankings directly. Same weight as those (see SOURCE_WEIGHTS).
    """
    return load_docx_rankings(path, "norris")


def load_dataroma(path: str) -> pd.DataFrame:
    """Dataroma's PPR redraft rankings. Uses Rank (overall), Tier, Player,
    Position, and "Pos Rank" (e.g. "RB1" - position letters glued to the
    rank number, split out here). Does not use the ADP/vs-ADP columns
    (that's Dataroma's own market comparison, not this project's).
    """
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "raw_name": raw["Player"],
        "position": raw["Position"],
        "dataroma_overall_rank": raw["Rank"].astype(int),
        "dataroma_tier": raw["Tier"].astype(int),
        "dataroma_position_rank": raw["Pos Rank"].str.replace(r"^[A-Za-z]+", "", regex=True).astype(int),
    })
    df["normalized_name"] = df["raw_name"].map(_normalize_name)
    return df


def load_barrett(path: str) -> pd.DataFrame:
    """Scott Barrett's overall redraft rankings (FantasyPoints). Scoring
    format assumed PPR (not stated in this export - see module docstring).
    Position rank is DERIVED (not published) by ranking within position by
    overall rank.
    """
    raw = pd.read_csv(path)
    df = pd.DataFrame({
        "raw_name": raw["NAME"],
        "position": raw["Position"],
        "barrett_overall_rank": raw["OVERALL"].astype(int),
    })
    df["barrett_position_rank"] = df.groupby("position")["barrett_overall_rank"].rank(method="first").astype(int)
    df["normalized_name"] = df["raw_name"].map(_normalize_name)
    return df


def load_hansen(path: str) -> pd.DataFrame:
    """John Hansen's top-200 overall redraft rankings (FantasyPoints).
    Explicitly PPR - verified via the export's own "Format" column.
    Position rank is DERIVED (not published) by ranking within position by
    overall rank.
    """
    raw = pd.read_csv(path)
    formats = raw["Format"].unique().tolist()
    if formats != ["ppr"]:
        print(f"  WARNING: hansen file has unexpected Format value(s): {formats} - expected only 'ppr'")
    df = pd.DataFrame({
        "raw_name": raw["NAME"],
        "position": raw["POS"],
        "hansen_overall_rank": raw["RANK"].astype(int),
    })
    df["hansen_position_rank"] = df.groupby("position")["hansen_overall_rank"].rank(method="first").astype(int)
    df["normalized_name"] = df["raw_name"].map(_normalize_name)
    return df


def add_position_pct(df: pd.DataFrame, rank_col: str, pct_col: str) -> pd.DataFrame:
    """Within-position percentile (rank / that source's own position pool
    size) - computed on the source's OWN full list, before any matching to
    our board, so a low match rate can't distort a player's true standing
    within that source's list.
    """
    df = df.copy()
    n_at_position = df.groupby("position")[rank_col].transform("count")
    df[pct_col] = df[rank_col] / n_at_position
    return df


def add_join_key(df: pd.DataFrame, id_col: str = "player_id", normalized_name_col: str = "normalized_name") -> pd.DataFrame:
    """A stable merge key that falls back to the normalized display name
    when player_id is null - the ~7 known 2026 UDFA rookies with no
    resolvable gsis_id (Carson Beck, Colbie Young, Oscar Delp, De'Zhaun
    Stribling, Nicholas Singleton, Joe Royer, Deion Burks - see CLAUDE.md).

    Using the raw (null) player_id as a merge key was silently discarding
    these players' real external-source rankings even when a source clearly
    ranked them - match_to_board's own name-based match succeeded (their
    name matched fine), but the row still carried player_id=NaN forward,
    and a later `.dropna(subset=["player_id"])` (needed elsewhere to avoid
    the NaN-merge-fan-out bug this project has hit repeatedly) threw it away
    regardless. Confirmed concretely: Dataroma/Barrett/Hansen/CSI all rank
    De'Zhaun Stribling in a normal, real range (top ~150), but he showed up
    at composite rank 734 (effectively last) purely from this join failure,
    not from the sources actually rating him that low.

    Not used as a replacement for player_id anywhere outside this module -
    it's a local, composite-board-only join convenience, not a claim that
    these players now have a real gsis_id.
    """
    df = df.copy()
    df["join_key"] = df[id_col].fillna(df[normalized_name_col])
    return df


def build_board_crosswalk(board: pd.DataFrame) -> pd.DataFrame:
    """normalized_name -> player_id/join_key/position lookup from OUR OWN
    already-built draft board - the composite board is anchored to this
    player set (see module docstring). Deduplicated by normalized_name (same
    precedent/caveat as paid_data.build_name_crosswalk: rare name
    collisions aren't resolved, first match wins).
    """
    cw = board[["player_id", "player_display_name", "position"]].copy()
    cw["normalized_name"] = cw["player_display_name"].map(_normalize_name)
    cw = add_join_key(cw)
    return cw.drop_duplicates(subset="normalized_name", keep="first")


def match_to_board(source: pd.DataFrame, crosswalk: pd.DataFrame, source_label: str, fuzzy_cutoff: float = 0.82) -> pd.DataFrame:
    """Match a parsed external source onto our board's join_key: exact
    normalized-name match first (name only, matching this project's
    existing paid-data precedent), then a FUZZY (difflib) fallback
    RESTRICTED TO THE SAME POSITION for anything left unmatched - covers
    real, verified name corruption (the CSI PDF's dropped ligatures) without
    risking a cross-position false positive, which is where fuzzy matching's
    real risk lives. Reports the real match rate and lists unmatched names,
    rather than assuming the merge worked. Matches (and reports) on
    join_key, not raw player_id - see add_join_key's docstring for why.
    """
    merged = source.merge(crosswalk[["normalized_name", "player_id", "join_key"]], on="normalized_name", how="left")

    fuzzy_hits = 0
    unmatched_mask = merged["join_key"].isna()
    for pos in merged.loc[unmatched_mask, "position"].unique():
        pool = crosswalk[crosswalk["position"] == pos]
        choices = pool["normalized_name"].tolist()
        name_to_key = dict(zip(pool["normalized_name"], pool["join_key"]))
        name_to_id = dict(zip(pool["normalized_name"], pool["player_id"]))
        rows = merged[unmatched_mask & (merged["position"] == pos)]
        for idx, row in rows.iterrows():
            close = difflib.get_close_matches(row["normalized_name"], choices, n=1, cutoff=fuzzy_cutoff)
            if close:
                merged.loc[idx, "join_key"] = name_to_key[close[0]]
                merged.loc[idx, "player_id"] = name_to_id[close[0]]
                fuzzy_hits += 1

    matched_n = int(merged["join_key"].notna().sum())
    print(f"  {source_label}: {matched_n}/{len(merged)} matched to our board ({fuzzy_hits} via fuzzy fallback)")
    still_missing = merged.loc[merged["join_key"].isna(), "raw_name"].tolist()
    if still_missing:
        shown = still_missing[:15]
        suffix = f" ... (+{len(still_missing) - 15} more)" if len(still_missing) > 15 else ""
        print(f"    unmatched ({len(still_missing)}): {shown}{suffix}")

    return merged.dropna(subset=["join_key"]).drop_duplicates(subset="join_key", keep="first")


# User's explicit choice (2026-08-31, revised same day - "deweight our model
# by half again"): our own model counts a QUARTER as much as Dataroma/
# Barrett/Hansen, CSI counts HALF as much. Two different, specific reasons,
# not the same rationale applied twice: CSI is a single analyst's coarser
# positional-only tiered read (no overall rank), while our own model is
# still walk-forward validating and the user specifically has low
# confidence in ITS durability estimates for players with a fresh, real
# opportunity change (new team/role) that a backward-looking games_est
# can't fully see yet - Kenneth Walker III and Jaylen Waddle cited as
# concrete examples. Weighting our own model down keeps the Consensus view
# an actual outside check rather than implicitly being mostly "us".
# Joel Smyth and Hayden Winks added 2026-09-01, same weight as Dataroma/
# Barrett/Hansen per the user's explicit instruction. Jeff Bell, Sigmund
# Bloom, and Josh Norris added 2026-09-02 - same weight again, bringing the
# total to 10 sources (our model + 9 external), per the user's explicit
# "get to 10" ask.
SOURCE_WEIGHTS = {"our": 0.25, "dataroma": 1.0, "barrett": 1.0, "hansen": 1.0, "csi": 0.5,
                  "smyth": 1.0, "winks": 1.0, "bell": 1.0, "bloom": 1.0, "norris": 1.0}


def weighted_mean(df: pd.DataFrame, cols: list[str], weights: list[float]) -> pd.Series:
    """Weighted row-wise average across `cols`, skipping null values the
    same way pandas' own `.mean(skipna=True)` does - a null value's weight
    is excluded from the denominator too, not just its contribution to the
    numerator, so a player ranked by only 2 of 5 sources isn't penalized
    for the other 3's absence.
    """
    values = df[cols]
    w = pd.Series(weights, index=cols)
    mask = values.notna()
    numerator = values.fillna(0).mul(w, axis=1).sum(axis=1)
    denominator = mask.mul(w, axis=1).sum(axis=1)
    return numerator / denominator


def build_composite_board(
    board: pd.DataFrame, dataroma: pd.DataFrame, barrett: pd.DataFrame, hansen: pd.DataFrame,
    csi: pd.DataFrame, smyth: pd.DataFrame, winks: pd.DataFrame,
    bell: pd.DataFrame, bloom: pd.DataFrame, norris: pd.DataFrame,
) -> pd.DataFrame:
    """Blend our own board with the 9 external sources - see module
    docstring for the full methodology and the two-composite-number design.
    Weighted per SOURCE_WEIGHTS, not a plain average - see that constant.
    QB/RB/WR/TE only (this is anchored to OUR board's player set, which
    doesn't cover K/DST at all) - Bell/Bloom/Norris's own K/DST rows simply
    fail to match anything in the crosswalk and fall out via match_to_board's
    existing dropna, same as any other unmatched row; see
    build_kdst_consensus_board for the separate K/DST treatment.
    """
    crosswalk = build_board_crosswalk(board)

    out = board.copy()
    out["normalized_name"] = out["player_display_name"].map(_normalize_name)
    out = add_join_key(out)
    out["our_overall_rank"] = out["vbd"].rank(ascending=False, method="min")
    out["our_position_pct"] = out["position_rank"] / out.groupby("position")["position_rank"].transform("count")

    dataroma = add_position_pct(dataroma, "dataroma_position_rank", "dataroma_position_pct")
    barrett = add_position_pct(barrett, "barrett_position_rank", "barrett_position_pct")
    hansen = add_position_pct(hansen, "hansen_position_rank", "hansen_position_pct")
    csi = add_position_pct(csi, "csi_position_rank", "csi_position_pct")
    smyth = add_position_pct(smyth, "smyth_position_rank", "smyth_position_pct")
    winks = add_position_pct(winks, "winks_position_rank", "winks_position_pct")
    bell = add_position_pct(bell, "bell_position_rank", "bell_position_pct")
    bloom = add_position_pct(bloom, "bloom_position_rank", "bloom_position_pct")
    norris = add_position_pct(norris, "norris_position_rank", "norris_position_pct")

    matched = {
        "dataroma": (match_to_board(dataroma, crosswalk, "Dataroma"),
                     ["join_key", "dataroma_overall_rank", "dataroma_tier", "dataroma_position_rank", "dataroma_position_pct"]),
        "barrett": (match_to_board(barrett, crosswalk, "Barrett"),
                    ["join_key", "barrett_overall_rank", "barrett_position_rank", "barrett_position_pct"]),
        "hansen": (match_to_board(hansen, crosswalk, "Hansen"),
                   ["join_key", "hansen_overall_rank", "hansen_position_rank", "hansen_position_pct"]),
        "csi": (match_to_board(csi, crosswalk, "CSI"),
                ["join_key", "csi_position_rank", "csi_tier", "csi_position_pct"]),
        "smyth": (match_to_board(smyth, crosswalk, "Smyth"),
                  ["join_key", "smyth_overall_rank", "smyth_position_rank", "smyth_position_pct"]),
        "winks": (match_to_board(winks, crosswalk, "Winks"),
                  ["join_key", "winks_overall_rank", "winks_position_rank", "winks_position_pct"]),
        "bell": (match_to_board(bell, crosswalk, "Bell"),
                 ["join_key", "bell_overall_rank", "bell_position_rank", "bell_position_pct"]),
        "bloom": (match_to_board(bloom, crosswalk, "Bloom"),
                  ["join_key", "bloom_overall_rank", "bloom_position_rank", "bloom_position_pct"]),
        "norris": (match_to_board(norris, crosswalk, "Norris"),
                   ["join_key", "norris_overall_rank", "norris_position_rank", "norris_position_pct"]),
    }
    for _, (df, cols) in matched.items():
        out = out.merge(df[cols], on="join_key", how="left")
    out = out.drop(columns=["join_key", "normalized_name"])

    pct_cols = ["our_position_pct", "dataroma_position_pct", "barrett_position_pct", "hansen_position_pct",
                "csi_position_pct", "smyth_position_pct", "winks_position_pct", "bell_position_pct",
                "bloom_position_pct", "norris_position_pct"]
    pct_weights = [SOURCE_WEIGHTS[n] for n in
                   ["our", "dataroma", "barrett", "hansen", "csi", "smyth", "winks", "bell", "bloom", "norris"]]
    out["consensus_position_pct"] = weighted_mean(out, pct_cols, pct_weights)
    out["n_sources"] = out[pct_cols].notna().sum(axis=1)

    overall_cols = ["our_overall_rank", "dataroma_overall_rank", "barrett_overall_rank", "hansen_overall_rank",
                    "smyth_overall_rank", "winks_overall_rank", "bell_overall_rank", "bloom_overall_rank",
                    "norris_overall_rank"]
    overall_weights = [SOURCE_WEIGHTS[n] for n in
                        ["our", "dataroma", "barrett", "hansen", "smyth", "winks", "bell", "bloom", "norris"]]
    out["consensus_overall_rank"] = weighted_mean(out, overall_cols, overall_weights)

    return out.sort_values("consensus_overall_rank", na_position="last").reset_index(drop=True)


def build_kdst_consensus_board(bell: pd.DataFrame, bloom: pd.DataFrame, winks: pd.DataFrame, norris: pd.DataFrame) -> pd.DataFrame:
    """Standalone K/DST consensus board - our own model doesn't project
    kickers or defenses at all, so there's no board to anchor to the way
    build_composite_board anchors QB/RB/WR/TE to our own model's player set
    (see module docstring). Built instead as a union of every K/DST player
    ANY of these 4 sources ranks (Bell/Bloom: full 32/32 K/DST coverage
    each; Winks: partial, 18 K/24 DST; Norris: 1 each, negligible but real)
    - not anchored to one source's own list, since there's no reason to
    drop a kicker Bloom ranks just because Bell doesn't happen to.

    DST identity is canonicalized by TEAM CODE, not raw name - sources
    disagree on how they write a defense's name (Bell/Bloom: "Miami
    Dolphins"; Winks/Norris: "MIA"), which would otherwise scatter the same
    real defense across multiple unmatched rows. Kicker identity still uses
    normalized player name (no such format disagreement observed for K).
    """
    sources = {"bell": bell, "bloom": bloom, "winks": winks, "norris": norris}
    kdst: dict[str, pd.DataFrame] = {}
    for name, df in sources.items():
        d = df[df["position"].isin(["K", "DST"])].copy()
        d["team"] = d["team"].str.upper().replace(_KDST_TEAM_FIXES)
        d["entity_key"] = d["normalized_name"].map(_strip_accents)
        dst_mask = d["position"] == "DST"
        d.loc[dst_mask, "entity_key"] = "dst_" + d.loc[dst_mask, "team"]
        d = add_position_pct(d, f"{name}_position_rank", f"{name}_position_pct")
        kdst[name] = d

    master = pd.concat([d[["entity_key", "raw_name", "position", "team"]] for d in kdst.values()], ignore_index=True)
    # Bell's xlsx export has a real mojibake byte in at least one name (Eddy
    # Piñeiro -> "Eddy Pi�eiro") - a genuine source-data corruption, not
    # a normalization bug (confirmed: _strip_accents already correctly
    # dedupes this player onto one entity_key; only the DISPLAY name is
    # broken). Prefer a source's clean spelling when one exists, rather than
    # always keeping the first (Bell-priority) row - a stable sort so every
    # other player's normal Bell-first preference is unaffected.
    master["_mojibake"] = master["raw_name"].str.contains("�", na=False)
    master = master.sort_values("_mojibake", kind="stable").drop_duplicates(subset="entity_key", keep="first")
    master = master.drop(columns="_mojibake").reset_index(drop=True)

    out = master.copy()
    for name, d in kdst.items():
        cols = ["entity_key", f"{name}_overall_rank", f"{name}_position_rank", f"{name}_position_pct"]
        out = out.merge(d[cols], on="entity_key", how="left")

    pct_cols = [f"{n}_position_pct" for n in kdst]
    pct_weights = [SOURCE_WEIGHTS[n] for n in kdst]
    out["consensus_position_pct"] = weighted_mean(out, pct_cols, pct_weights)
    out["n_sources"] = out[pct_cols].notna().sum(axis=1)

    overall_cols = [f"{n}_overall_rank" for n in kdst]
    overall_weights = [SOURCE_WEIGHTS[n] for n in kdst]
    out["consensus_overall_rank"] = weighted_mean(out, overall_cols, overall_weights)

    out["player_display_name"] = out["raw_name"]
    return out.drop(columns=["entity_key"]).sort_values("consensus_position_pct", na_position="last").reset_index(drop=True)

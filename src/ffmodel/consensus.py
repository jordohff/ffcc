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

Design choices, and why:
- EQUAL WEIGHT across sources (including our own model), per the user's
  explicit choice - no source gets more say than another.
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

import pandas as pd

from .paid_data import _normalize_name

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


def build_board_crosswalk(board: pd.DataFrame) -> pd.DataFrame:
    """normalized_name -> player_id/position lookup from OUR OWN already-
    built draft board - the composite board is anchored to this player set
    (see module docstring). Deduplicated by normalized_name (same
    precedent/caveat as paid_data.build_name_crosswalk: rare name
    collisions aren't resolved, first match wins).
    """
    cw = board[["player_id", "player_display_name", "position"]].copy()
    cw["normalized_name"] = cw["player_display_name"].map(_normalize_name)
    return cw.drop_duplicates(subset="normalized_name", keep="first")


def match_to_board(source: pd.DataFrame, crosswalk: pd.DataFrame, source_label: str, fuzzy_cutoff: float = 0.82) -> pd.DataFrame:
    """Match a parsed external source onto our board's player_id: exact
    normalized-name match first (name only, matching this project's
    existing paid-data precedent), then a FUZZY (difflib) fallback
    RESTRICTED TO THE SAME POSITION for anything left unmatched - covers
    real, verified name corruption (the CSI PDF's dropped ligatures) without
    risking a cross-position false positive, which is where fuzzy matching's
    real risk lives. Reports the real match rate and lists unmatched names,
    rather than assuming the merge worked.
    """
    merged = source.merge(crosswalk[["normalized_name", "player_id"]], on="normalized_name", how="left")

    fuzzy_hits = 0
    unmatched_mask = merged["player_id"].isna()
    for pos in merged.loc[unmatched_mask, "position"].unique():
        pool = crosswalk[crosswalk["position"] == pos]
        choices = pool["normalized_name"].tolist()
        name_to_id = dict(zip(pool["normalized_name"], pool["player_id"]))
        rows = merged[unmatched_mask & (merged["position"] == pos)]
        for idx, row in rows.iterrows():
            close = difflib.get_close_matches(row["normalized_name"], choices, n=1, cutoff=fuzzy_cutoff)
            if close:
                merged.loc[idx, "player_id"] = name_to_id[close[0]]
                fuzzy_hits += 1

    matched_n = int(merged["player_id"].notna().sum())
    print(f"  {source_label}: {matched_n}/{len(merged)} matched to our board ({fuzzy_hits} via fuzzy fallback)")
    still_missing = merged.loc[merged["player_id"].isna(), "raw_name"].tolist()
    if still_missing:
        shown = still_missing[:15]
        suffix = f" ... (+{len(still_missing) - 15} more)" if len(still_missing) > 15 else ""
        print(f"    unmatched ({len(still_missing)}): {shown}{suffix}")

    return merged.dropna(subset=["player_id"]).drop_duplicates(subset="player_id", keep="first")


def build_composite_board(board: pd.DataFrame, dataroma: pd.DataFrame, barrett: pd.DataFrame, hansen: pd.DataFrame, csi: pd.DataFrame) -> pd.DataFrame:
    """Blend our own board with the 4 external sources - see module
    docstring for the full methodology and the two-composite-number design.
    """
    crosswalk = build_board_crosswalk(board)

    out = board.copy()
    out["our_overall_rank"] = out["vbd"].rank(ascending=False, method="min")
    out["our_position_pct"] = out["position_rank"] / out.groupby("position")["position_rank"].transform("count")

    dataroma = add_position_pct(dataroma, "dataroma_position_rank", "dataroma_position_pct")
    barrett = add_position_pct(barrett, "barrett_position_rank", "barrett_position_pct")
    hansen = add_position_pct(hansen, "hansen_position_rank", "hansen_position_pct")
    csi = add_position_pct(csi, "csi_position_rank", "csi_position_pct")

    matched = {
        "dataroma": (match_to_board(dataroma, crosswalk, "Dataroma"),
                     ["player_id", "dataroma_overall_rank", "dataroma_tier", "dataroma_position_rank", "dataroma_position_pct"]),
        "barrett": (match_to_board(barrett, crosswalk, "Barrett"),
                    ["player_id", "barrett_overall_rank", "barrett_position_rank", "barrett_position_pct"]),
        "hansen": (match_to_board(hansen, crosswalk, "Hansen"),
                   ["player_id", "hansen_overall_rank", "hansen_position_rank", "hansen_position_pct"]),
        "csi": (match_to_board(csi, crosswalk, "CSI"),
                ["player_id", "csi_position_rank", "csi_tier", "csi_position_pct"]),
    }
    for _, (df, cols) in matched.items():
        out = out.merge(df[cols], on="player_id", how="left")

    pct_cols = ["our_position_pct", "dataroma_position_pct", "barrett_position_pct", "hansen_position_pct", "csi_position_pct"]
    out["consensus_position_pct"] = out[pct_cols].mean(axis=1, skipna=True)
    out["n_sources"] = out[pct_cols].notna().sum(axis=1)

    overall_cols = ["our_overall_rank", "dataroma_overall_rank", "barrett_overall_rank", "hansen_overall_rank"]
    out["consensus_overall_rank"] = out[overall_cols].mean(axis=1, skipna=True)

    return out.sort_values("consensus_overall_rank", na_position="last").reset_index(drop=True)

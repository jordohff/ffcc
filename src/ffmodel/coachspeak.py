"""Ingestion for The Coachspeak Index's compiled coach-presser notes (Greg
Brainos, data/paid/Coachspeak/ - hand-provided, gitignored, same treatment
as paid_data.py's FantasyPoints CSVs since this is proprietary third-party
data, not something a script can re-fetch).

The source file is a raw Discord channel export (.docx), 32 team sections in
a fixed alphabetical order (verified against "Coach Gauges.docx", a separate
screenshot doc from the same site - not parsed here, since its dial values
have no numeric labels and are strictly less precise than the dated bracket
annotations found inline in this document), each internally chronological
Jan-Aug 2026. There is no explicit team-header text anywhere in the export -
team boundaries are detected structurally, as a backward jump in the post
timestamp (confirmed empirically: exactly 31 backward jumps for 32 sections,
every one landing between a late-August date and an early-year date).

Deliberately informational-only (see the coachspeak-plan memory) - this
module attaches recent, dated coach quotes and each coach's self-reported
reliability score to affected players on the board, but does NOT feed any
of it into ppg_pred/games_est/VBD. Only ~7 months of reliability-score
history exist (this site's first season of coverage, per the user) -
nowhere near enough to walk-forward validate a numeric weighting scheme the
way every other correction in this pipeline has been validated, so this
stays a pure overlay until multiple seasons accumulate.
"""

import re
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd

W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

# Alphabetical-by-city team order, confirmed both from "Coach Gauges.docx"
# (visually read, one screenshot per team in this order) and structurally
# from this document's own 31 backward-timestamp-jumps landing at exactly
# these 32 section boundaries.
TEAM_SECTION_ORDER = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LV", "LAC", "LA", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SF", "SEA", "TB",
    "TEN", "WAS",
]

_ENTRY_START_RE = re.compile(r"^Greg Brainos \[CSI\]")
_TIMESTAMP_RE = re.compile(r"(\d{1,2}/\d{1,2}/\d{4})\s+\d{1,2}:\d{2}\s*[AP]M")
_CONTENT_DATE_RE = re.compile(r"^(\d{2}/\d{2}/\d{2})\s*(.*)$")
_RELATIVE_TS_RE = re.compile(r"^(Today|Yesterday) at \d{1,2}:\d{2}\s*[AP]M$")
_COACH_LINE_RE = re.compile(
    r"^(HC|OC|DC|GM|EVP|de facto GM|QB Coach)\s+([A-Z][\w.'’\-]+(?:\s+[A-Z][\w.'’\-]+){0,3})"
)
_SUFFIX_RE = re.compile(r"\s+(Jr\.?|Sr\.?|II|III|IV)$", re.IGNORECASE)

RELIABILITY_CATEGORY_MAP = {
    "injury": "injury",
    "injuries": "injury",
    "depth chart": "depth_chart",
    "usage/workload": "usage_workload",
    "usage": "usage_workload",
    "workload": "usage_workload",
    "transaction": "transactions",
    "transactions": "transactions",
}
_CATEGORY_ALTS = "|".join(sorted(RELIABILITY_CATEGORY_MAP, key=len, reverse=True))

_RATING_RE = re.compile(
    rf"\[(?P<name>[A-Z][\w.'’\-]*(?:\s+[A-Z][\w.'’\-]*)*) has (?:a|an) (?P<score>\d+)% "
    rf"reliability rating on (?P<category>{_CATEGORY_ALTS}) coachspeak",
    re.IGNORECASE,
)
_UPDATE_PCT_RE = re.compile(
    rf"(?P<category>{_CATEGORY_ALTS}) coachspeak (?P<old>\d+)%(?:\s*\([^)]*\))?\s+"
    rf"(?P<new>\d+)%(?:\s*\([^)]*\))?",
    re.IGNORECASE,
)
_UPDATE_BARE_RE = re.compile(
    rf"(?P<category>{_CATEGORY_ALTS})\s+(?P<old>\d+)\s+(?P<new>\d+)(?!%)",
    re.IGNORECASE,
)
_BRACKET_RE = re.compile(r"\[([^\[\]]*)\]")


def _extract_docx_paragraphs(docx_path: str) -> list[str]:
    """One paragraph (<w:p>) per list entry, in document order - preserves
    blank paragraphs, which matter here as structural spacing between
    sub-quotes within an entry.
    """
    with zipfile.ZipFile(docx_path) as z:
        xml_bytes = z.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    paragraphs = []
    for p in root.iter(f"{W_NS}p"):
        texts = [t.text or "" for t in p.iter(f"{W_NS}t")]
        paragraphs.append("".join(texts))
    return paragraphs


def parse_coachspeak_entries(docx_path: str) -> pd.DataFrame:
    """Parse the Discord export into one row per real presser note.

    Columns: team, date (the presser's own date, MM/DD/YY - falls back to
    the Discord post date if no content-date line was found), coaches (list
    of "TITLE Name" strings mentioned in this entry's header - may be more
    than one for a joint presser), body (the quote/paraphrase text).

    Boilerplate "Greg Brainos pinned a message..." system notices and any
    entry whose body is empty after removing them are dropped.
    """
    paragraphs = _extract_docx_paragraphs(docx_path)
    n = len(paragraphs)

    # Locate every real entry-start paragraph (requires "[CSI]" so the
    # unrelated "Greg Brainos pinned a message..." notice, which starts a
    # separate paragraph with just "Greg Brainos" and no "[CSI]", doesn't
    # get mistaken for a new entry boundary).
    starts = [i for i, p in enumerate(paragraphs) if _ENTRY_START_RE.match(p)]

    rows = []
    team_idx = 0
    prev_date = None
    for k, start in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else n
        block = paragraphs[start:end]

        # Timestamp: search the start paragraph + next couple for "M/D/YYYY H:MM AM/PM".
        post_date = None
        for p in block[:3]:
            m = _TIMESTAMP_RE.search(p)
            if m:
                post_date = pd.Timestamp(m.group(1))
                break

        # Team-section boundary: a backward jump in the post timestamp means
        # we've crossed into the next team's section.
        if post_date is not None:
            if prev_date is not None and post_date < prev_date:
                team_idx = min(team_idx + 1, len(TEAM_SECTION_ORDER) - 1)
            prev_date = post_date

        # Body candidate lines: everything in the block after the header
        # timestamp lines, skipping blanks and the pinned-message boilerplate.
        content_lines = [
            p.strip() for p in block[1:]
            if p.strip() and "pinned a message" not in p and p.strip() not in ("—", "Greg Brainos")
            and not _TIMESTAMP_RE.fullmatch(p.strip()) and not _RELATIVE_TS_RE.match(p.strip())
        ]
        if not content_lines:
            continue
        # The one-time formatting-note entry at the very top of the doc
        # ("A QUICK NOTE ON FORMATTING...") isn't a real presser - skip it.
        if "QUICK NOTE ON FORMATTING" in content_lines[0]:
            continue

        # First line is normally "MM/DD/YY" or "MM/DD/YY <rest>".
        content_date = post_date
        idx = 0
        m = _CONTENT_DATE_RE.match(content_lines[0])
        if m:
            try:
                content_date = pd.to_datetime(m.group(1), format="%m/%d/%y")
            except ValueError:
                pass
            content_lines[0] = m.group(2).strip()
            if not content_lines[0]:
                idx = 1

        # Collect coach title/name lines immediately following the date.
        coaches = []
        while idx < len(content_lines):
            cm = _COACH_LINE_RE.match(content_lines[idx])
            if not cm:
                break
            coaches.append(f"{cm.group(1)} {cm.group(2)}")
            idx += 1

        body = "\n".join(content_lines[idx:]).strip()
        if not body:
            continue

        rows.append(
            {
                "team": TEAM_SECTION_ORDER[team_idx],
                "date": content_date,
                "coaches": "; ".join(coaches),
                "body": body,
            }
        )

    return pd.DataFrame(rows)


def extract_reliability_scores(entries: pd.DataFrame) -> pd.DataFrame:
    """Scan every entry's body for Coachspeak Index bracket annotations and
    return a long-format table: team, date, category, score, low_sample.

    Two real bracket shapes found in the source (verified against all 30
    bracket-with-percent annotations in the 8/30/26 file before writing
    these patterns, not guessed): "[Name has an 85% reliability rating on
    injury coachspeak...]" (current score, name-attributed) and
    "[injury coachspeak 60% -> 65%]" / "[depth chart 20 -> 35  usage/
    workload 60 -> 70]" (an old->new update, sometimes bare numbers with no
    "coachspeak"/"%" at all, and the arrow character itself doesn't survive
    plaintext extraction - it collapses to whitespace, which the regexes
    below treat as the separator). Update-only rows resolve to the row's own
    team (no name given) since they always appear within a single team's
    section.
    """
    rows = []
    for _, row in entries.iterrows():
        for bracket in _BRACKET_RE.findall(row["body"]):
            text = "[" + bracket + "]"
            low_sample = "sample size" in bracket.lower()

            m = _RATING_RE.search(text)
            if m:
                rows.append(
                    {
                        "team": row["team"], "date": row["date"], "coach_name": m.group("name"),
                        "category": RELIABILITY_CATEGORY_MAP[m.group("category").lower()],
                        "score": int(m.group("score")), "low_sample": low_sample,
                    }
                )
                continue

            for m in _UPDATE_PCT_RE.finditer(text):
                rows.append(
                    {
                        "team": row["team"], "date": row["date"], "coach_name": None,
                        "category": RELIABILITY_CATEGORY_MAP[m.group("category").lower()],
                        "score": int(m.group("new")), "low_sample": low_sample,
                    }
                )
            if _UPDATE_PCT_RE.search(text):
                continue

            for m in _UPDATE_BARE_RE.finditer(bracket):
                rows.append(
                    {
                        "team": row["team"], "date": row["date"], "coach_name": None,
                        "category": RELIABILITY_CATEGORY_MAP[m.group("category").lower()],
                        "score": int(m.group("new")), "low_sample": low_sample,
                    }
                )

    return pd.DataFrame(rows)


def latest_reliability_by_team(reliability: pd.DataFrame) -> pd.DataFrame:
    """Most recent score per team/category, wide - one row per team, columns
    reliability_injury/reliability_depth_chart/reliability_usage_workload/
    reliability_transactions. A team/category with no annotation anywhere in
    the source stays NaN (matches "insufficient data" on the gauges site).
    """
    if reliability.empty:
        return pd.DataFrame(columns=["team"])
    latest = (
        reliability.sort_values("date")
        .drop_duplicates(subset=["team", "category"], keep="last")
        .pivot(index="team", columns="category", values="score")
        .add_prefix("reliability_")
        .reset_index()
    )
    return latest


def find_player_quotes(entries: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Tag each entry's body with any of that TEAM's own rostered players
    mentioned by name. `roster` needs columns [player_id,
    player_display_name, team] (pass the board itself, pre-coachspeak-merge,
    filtered to non-null team) - matching is restricted to a player's own
    team's coach, deliberately: a coach talking about an opposing player is
    rare and not the relationship this overlay is meant to surface, and
    restricting this way avoids false positives from common surnames shared
    across teams.

    Returns one row per (entry, player) match: player_id, team, date,
    coaches, body.
    """
    roster = roster.dropna(subset=["team", "player_display_name"]).copy()
    # A name needs both a first and last token to search for safely (a
    # bare last name alone risks matching unrelated text); build a
    # case-sensitive "First Last" regex per player from their own
    # (suffix-stripped) display name.
    roster["_search_name"] = roster["player_display_name"].map(
        lambda n: _SUFFIX_RE.sub("", str(n)).strip()
    )
    roster = roster[roster["_search_name"].str.split().str.len() >= 2]

    by_team: dict[str, pd.DataFrame] = {t: g for t, g in roster.groupby("team")}

    rows = []
    for _, row in entries.iterrows():
        team_roster = by_team.get(row["team"])
        if team_roster is None:
            continue
        # Entries are multi-topic (several "on X: ..." units per post) - a
        # blanket truncation of the whole body would often cut off before
        # reaching the sentence that actually mentions the player. Split on
        # the paragraph boundaries already present (parse_coachspeak_entries
        # joins each entry's paragraphs with "\n") and keep only the ones
        # that actually mention the player, so the stored snippet is always
        # on-topic even for a long, multi-subject presser.
        paragraphs = row["body"].split("\n")
        for _, prow in team_roster.iterrows():
            pattern = r"\b" + re.escape(prow["_search_name"]) + r"\b"
            hits = [p for p in paragraphs if re.search(pattern, p)]
            if hits:
                rows.append(
                    {
                        "player_id": prow["player_id"], "team": row["team"], "date": row["date"],
                        "coaches": row["coaches"], "snippet": " ".join(hits),
                    }
                )

    return pd.DataFrame(rows)


def build_coachspeak_overlay(
    quotes: pd.DataFrame, reliability_by_team: pd.DataFrame, max_quotes: int = 2, snippet_len: int = 280
) -> pd.DataFrame:
    """One row per player_id with up to `max_quotes` most recent tagged
    quotes (formatted "MM/DD/YY (coaches): snippet") joined by " || ", the
    date of the most recent one, and that player's TEAM's current 4
    reliability scores broadcast onto every player on that team. Pure
    informational columns - merge onto the board with how="left"; a player
    with no quotes gets NaN throughout, not an error.
    """
    if quotes.empty:
        return pd.DataFrame(columns=["player_id", "coachspeak_notes", "coachspeak_last_date"])

    quotes = quotes.sort_values("date", ascending=False)

    def _format(group: pd.DataFrame) -> pd.Series:
        top = group.head(max_quotes)
        parts = []
        for _, r in top.iterrows():
            snippet = r["snippet"]
            if len(snippet) > snippet_len:
                snippet = snippet[:snippet_len].rsplit(" ", 1)[0] + "..."
            date_str = r["date"].strftime("%m/%d/%y") if pd.notna(r["date"]) else "?"
            coach = f" ({r['coaches']})" if r["coaches"] else ""
            parts.append(f"{date_str}{coach}: {snippet}")
        last_date = group["date"].max()
        return pd.Series(
            {
                "coachspeak_notes": " || ".join(parts),
                # A plain date string (not a Timestamp) so this survives a
                # blanket board.round(2) on the final board unremarked -
                # pandas warns on rounding a datetime column - and reads
                # cleanly in the CSV alongside the individual quote dates.
                "coachspeak_last_date": last_date.strftime("%m/%d/%y") if pd.notna(last_date) else None,
            }
        )

    overlay = quotes.groupby("player_id", as_index=False).apply(_format, include_groups=False)
    overlay = overlay.merge(
        quotes[["player_id", "team"]].drop_duplicates("player_id"), on="player_id", how="left"
    )
    overlay = overlay.merge(reliability_by_team, on="team", how="left").drop(columns=["team"])
    return overlay

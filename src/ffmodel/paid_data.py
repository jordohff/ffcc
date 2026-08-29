"""Loaders for hand-downloaded paid third-party data (currently FantasyPoints
weekly export reports) - kept separate from data.py (which is exclusively
programmatic nflreadpy/Sleeper pulls) since these are manually downloaded
CSVs dropped in data/paid/ (gitignored - see .gitignore for why) rather than
something a script can re-fetch.

FantasyPoints export quirks handled here, verified against real files
(2026-08-28):
- Wide format: one row per player (or team, for the PROE report) per season,
  with separate W1-W18 columns instead of one row per player-week - melted
  into long format by _melt_weekly_report.
- Team codes differ from nflverse's convention for 4 teams (ARZ/BLT/CLV/HST
  vs standard ARI/BAL/CLE/HOU) - see FANTASYPOINTS_TEAM_FIXES.
- A player who changed teams mid-season shows as e.g. "ARZ, DEN" in the Team
  column - only the FIRST listed team is kept (a real, minor simplification:
  this loses which specific weeks were with which team, but the alternative
  - not using these players' data at all - throws away more).
- No player_id/gsis_id column at all, only a plain display name, WITH
  suffixes (Jr./Sr./II/III/IV) already stripped (confirmed: "Marvin Harrison
  Jr." appears as "Marvin Harrison") - build_name_crosswalk strips the same
  suffixes from nflreadpy's player_display_name before matching, and reports
  the real match rate rather than assuming it's complete.
"""

import re

import pandas as pd

FANTASYPOINTS_TEAM_FIXES = {"ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU"}

WEEK_COLUMNS = [f"W{i}" for i in range(1, 19)]

_SUFFIX_RE = re.compile(r"\s+(Jr\.?|Sr\.?|II|III|IV)$", re.IGNORECASE)


def _normalize_name(name: str) -> str:
    """Strip suffixes (Jr./Sr./II/III/IV) and punctuation, lowercase - the
    same transformation FantasyPoints' own export already applies to names,
    so nflreadpy's player_display_name needs the same treatment to match.
    """
    name = _SUFFIX_RE.sub("", str(name)).strip()
    name = name.replace(".", "").replace("'", "")
    return name.lower()


def build_name_crosswalk(weekly_stats: pd.DataFrame, season: int) -> pd.DataFrame:
    """Build a normalized-name -> player_id lookup from this project's own
    weekly_stats for the given season, for matching against FantasyPoints'
    suffix-stripped names. Returns columns [normalized_name, player_id,
    player_display_name, position] - deduplicated, keeping the first match
    for any name collision (rare; not attempting to resolve them here).
    """
    w = weekly_stats[weekly_stats["season"] == season][
        ["player_id", "player_display_name", "position"]
    ].drop_duplicates(subset="player_id")
    w = w.copy()
    w["normalized_name"] = w["player_display_name"].map(_normalize_name)
    return w.drop_duplicates(subset="normalized_name", keep="first")


def _melt_weekly_report(df: pd.DataFrame, id_cols: list[str], value_name: str) -> pd.DataFrame:
    """Melt a FantasyPoints wide weekly export (W1-W18 columns) into one row
    per id/week, dropping bye/blank weeks (empty string in the source).
    """
    present_weeks = [c for c in WEEK_COLUMNS if c in df.columns]
    long = df.melt(id_vars=id_cols, value_vars=present_weeks, var_name="week", value_name=value_name)
    long["week"] = long["week"].str.replace("W", "", regex=False).astype(int)
    long[value_name] = pd.to_numeric(long[value_name], errors="coerce")
    return long.dropna(subset=[value_name])


def _first_team(team_field: str) -> str:
    """A player who changed teams mid-season shows as "ARZ, DEN" - keep only
    the first-listed team (see module docstring for the tradeoff).
    """
    return str(team_field).split(",")[0].strip()


def _load_player_report(path: str, value_col: str, value_name: str, crosswalk: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["team"] = raw["Team"].map(_first_team).replace(FANTASYPOINTS_TEAM_FIXES)
    raw["season"] = raw["Season"]
    raw["normalized_name"] = raw["Name"].map(_normalize_name)

    merged = raw.merge(crosswalk[["normalized_name", "player_id"]], on="normalized_name", how="left")
    matched = merged["player_id"].notna().mean()
    print(f"  {path}: {100*matched:.1f}% of {len(merged)} rows matched a player_id")

    long = _melt_weekly_report(
        merged.dropna(subset=["player_id"]),
        id_cols=["player_id", "team", "season", "POS"],
        value_name=value_name,
    )
    return long.rename(columns={"POS": "position"})


def load_paid_snap_share(path: str, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """FantasyPoints weekly offensive snap share (RB/WR/TE/FB), as a
    fraction (not percent) to match this project's own snap_share
    convention. Columns: player_id, team, season, position, week,
    fp_snap_share.
    """
    df = _load_player_report(path, "Snap %", "fp_snap_share", crosswalk)
    df["fp_snap_share"] = df["fp_snap_share"] / 100
    return df


def load_paid_route_share(path: str, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """FantasyPoints weekly route participation share (WR/TE only, "TM RTE
    %" - the player's share of the TEAM's routes run that week). Columns:
    player_id, team, season, position, week, fp_route_share.
    """
    df = _load_player_report(path, "TM RTE %", "fp_route_share", crosswalk)
    df["fp_route_share"] = df["fp_route_share"] / 100
    return df


def load_paid_target_share(path: str, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """FantasyPoints weekly target share (WR/TE only, "TM TGT %"). Columns:
    player_id, team, season, position, week, fp_target_share.
    """
    df = _load_player_report(path, "TM TGT %", "fp_target_share", crosswalk)
    df["fp_target_share"] = df["fp_target_share"] / 100
    return df


MASCOT_TO_TEAM = {
    "Cardinals": "ARI", "Falcons": "ATL", "Ravens": "BAL", "Bills": "BUF",
    "Panthers": "CAR", "Bears": "CHI", "Bengals": "CIN", "Browns": "CLE",
    "Cowboys": "DAL", "Broncos": "DEN", "Lions": "DET", "Packers": "GB",
    "Texans": "HOU", "Colts": "IND", "Jaguars": "JAX", "Chiefs": "KC",
    "Rams": "LA", "Chargers": "LAC", "Raiders": "LV", "Dolphins": "MIA",
    "Vikings": "MIN", "Patriots": "NE", "Saints": "NO", "Giants": "NYG",
    "Jets": "NYJ", "Eagles": "PHI", "Steelers": "PIT", "Seahawks": "SEA",
    "49ers": "SF", "Buccaneers": "TB", "Titans": "TEN", "Commanders": "WAS",
    "Football Team": "WAS",  # Washington's name before rebranding in 2022
}
"""Maps FantasyPoints' PROE report "Team Name" (mascot only, e.g. "Cardinals")
to this project's standard team code - mascots are unique across the league
so this doesn't need the city/location field too. Includes "Football Team"
for Washington's 2021 name, before their 2022 rebrand to "Commanders".
"""


def load_paid_proe(path: str) -> pd.DataFrame:
    """FantasyPoints weekly team-level Pass Rate Over Expectation. Columns:
    team, season, week, fp_proe. PROE is a signed value (points above/below
    expected pass rate given game situation), NOT a percent - not divided
    by 100, unlike the player-level share reports.
    """
    raw = pd.read_csv(path)
    raw["team"] = raw["Team Name"].map(MASCOT_TO_TEAM)
    unmatched = raw["team"].isna().sum()
    if unmatched:
        print(f"  {path}: {unmatched} team(s) failed to map - check MASCOT_TO_TEAM")
    raw["season"] = raw["Season"]
    long = _melt_weekly_report(raw.dropna(subset=["team"]), id_cols=["team", "season"], value_name="fp_proe")
    return long

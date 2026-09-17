# data/fallback/

Committed (not gitignored, unlike everything under `data/raw/`) last-known-good
snapshots used only when a live data source can't be reached.

- `sleeper_players_snapshot.parquet` — a copy of Sleeper's player list. Exists
  because the scheduled cloud routine's sandbox blocks outbound requests to
  `api.sleeper.app` at the network-policy level (a 403 from its egress proxy,
  confirmed 2026-09-16 — not transient, and not a per-repo setting exposed to
  us), while every other pulled source here (all nflreadpy/nflverse-backed)
  works fine from that same sandbox. `scripts/pull_data.py`'s
  `_pull_sleeper_players` falls back to this file when the live pull fails,
  and refreshes it automatically whenever a live pull *does* succeed (i.e.
  whenever someone with normal network access — a human session — runs the
  script). So it drifts only as stale as "since the last real refresh," not
  something that needs separate manual upkeep.

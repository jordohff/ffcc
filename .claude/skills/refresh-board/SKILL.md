---
name: refresh-board
description: On-demand mid-season refresh - re-pull live NFL data (depth charts, rosters, injuries, Sleeper), regenerate both draft boards, diff against the previous board for notable movers, verify anything suspicious with real web research, and summarize what changed. Use when the user asks to refresh, update, or re-check the board with current news/depth-chart/injury/snap-share changes.
---

# Refresh the draft board with current live data

This project's core value system (role-security discount, role-upgrade rate/durability corrections,
QB starter floor, the "returning from a lost season" path) is built entirely on LIVE, re-pullable
signals - `depth_chart_rank`, injury reports, current team/roster status. It only stays accurate if
that underlying data is fresh. This skill re-pulls the live sources and surfaces what changed, so
findings like the Deshaun Watson case (2026-08-27 - a depth chart update revealed a real Week 1 starter
who was completely missing from the board) get caught routinely instead of by accident.

Follow these steps in order. Do not skip the verification/summary steps even if the mechanical re-pull
looks uneventful - the whole point is surfacing what a human would otherwise have to notice by hand.

## 1. Snapshot the current board before touching anything

Copy the current `output/draft_rankings/draft_rankings_2026_half_ppr.csv` and
`draft_rankings_2026_ppr.csv` to the scratchpad (e.g. `prev_half_ppr.csv`, `prev_ppr.csv`) so you can
diff against them after regenerating. If they don't exist yet (first run), skip this step and skip the
diff in step 4.

## 2. Re-pull live data only

Run `uv run scripts/pull_data.py --refresh-live` (NOT `--force`, which re-pulls years of static
historical data unnecessarily). This refreshes exactly the sources that can change mid-season:
`sleeper_players.parquet`, `current_depth_chart.parquet`, `rosters.parquet`, `injuries.parquet`.

## 3. Regenerate both boards

`uv run scripts/build_draft_rankings.py --scoring half_ppr` and `--scoring ppr`. Note the backtest
numbers printed at the start of each run - if either position's Spearman/hit-rate moved meaningfully
from what's recorded in CLAUDE.md, investigate before trusting the rest of the output (a big backtest
swing from a live-data-only refresh would itself be a red flag, since backtest doesn't depend on the
live sources at all - it would suggest something upstream broke, not a real signal to report).

## 4. Diff against the previous board

**Drop rows with a null `player_id` from BOTH sides before joining.** This project has hit the same
NaN-merge-fan-out bug repeatedly (pandas treats NaN as matching NaN): a handful of UDFA rookies with no
resolvable gsis_id share `player_id = NaN`, and joining without dropping them first produces spurious
duplicate rows with wild, meaningless rank deltas. This is a known, deliberately-deferred issue (see
CLAUDE.md, 2026-08-09) - don't re-report it as a new finding each refresh.

For each scoring format, join old vs new on `player_id` (nulls dropped), and surface:
- Players whose `depth_chart_rank` changed (a real role change - promotion, benching, injury).
- Players who appeared or disappeared entirely (a new "returning from a lost season" case, a
  retirement/CUT/RET status change, or a genuinely new roster addition).
- Players whose overall rank moved by a large amount (e.g. 50+ spots) - the practical signal a human
  actually cares about.

## 5. Run the team-PPG-consistency check

Reuse the diagnostic built 2026-08-27 (see CLAUDE.md): for each team, compare the current
`depth_chart_rank==1` QB's `ppg_pred` against that team's combined RB/WR/TE `ppg_pred` sum. Flag any
team outside the historical normal ratio range (~4.06-8.39, skill_ppg_sum / qb1_ppg) for a closer look -
this is what caught the Deshaun Watson gap. A team with NO qualifying QB1 row at all is itself a red
flag (the exact shape of the Watson bug) and should be investigated immediately, not just logged.

## 6. Verify anything real with actual web research

For each notable change surfaced in steps 4-5 (not routine roster churn - use judgment), do a targeted
WebSearch/WebFetch check against real news before reporting it as fact, the same way the Watson/Hafley/
Willis situations were verified earlier this project. Don't just trust what the re-pulled data implies -
confirm it against a real source, especially for anything that would meaningfully move a notable
player's ranking.

## 7. Summarize plainly

Report: what was re-pulled, what changed (depth chart moves, new/removed players, notable rank swings),
what you verified via web research and what you found, and anything that still looks off and needs a
follow-up investigation (don't silently "fix" a new anomaly the same day you find it unless it's
clearly the same, already-validated class of issue - e.g. another literal missing-QB1 case is safe to
resolve the same way Watson was; a genuinely new pattern deserves the same research-before-shipping
discipline as everything else in this project).

## 8. Ask about git - every time, without exception

After presenting the summary, explicitly ask the user whether they want to commit the refreshed
board/data. This is a standing instruction (2026-08-27) - do not skip this step or assume the answer,
even if a previous refresh was committed. Do not commit without an explicit yes.

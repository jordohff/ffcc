"""Monte Carlo simulation for a specific drafted fantasy roster (phase 2).

Reuses the board and walk-forward residual pool already written by
build_draft_rankings.py - run that first (for the scoring format/season you
want) before this script.

Usage:
    uv run scripts/simulate_roster.py --players "Jahmyr Gibbs" "Ja'Marr Chase" "Josh Allen" ...
    uv run scripts/simulate_roster.py --roster-file my_team.txt
    uv run scripts/simulate_roster.py --roster-file my_team.txt --scoring ppr
"""

import argparse
from pathlib import Path

import pandas as pd

from ffmodel.season import simulate_roster_outcomes

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "draft_rankings"


def load_roster_names(args: argparse.Namespace) -> list[str]:
    if args.roster_file:
        lines = Path(args.roster_file).read_text().splitlines()
        return [line.strip() for line in lines if line.strip()]
    return args.players


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scoring", choices=["half_ppr", "ppr"], default="half_ppr")
    parser.add_argument("--draft-season", type=int, default=2026)
    parser.add_argument("--players", nargs="+", default=[], help="Player names, space-separated (quote multi-word names)")
    parser.add_argument("--roster-file", type=str, default=None, help="Path to a text file, one player name per line")
    parser.add_argument("--n-sims", type=int, default=10000)
    args = parser.parse_args()

    names = load_roster_names(args)
    if not names:
        parser.error("provide a roster via --players or --roster-file")

    board_path = OUTPUT_DIR / f"draft_rankings_{args.draft_season}_{args.scoring}.csv"
    residuals_path = OUTPUT_DIR / f"walk_forward_residuals_{args.scoring}.csv"
    if not board_path.exists() or not residuals_path.exists():
        raise SystemExit(
            f"Missing {board_path} or {residuals_path} - run "
            f"`uv run scripts/build_draft_rankings.py --scoring {args.scoring} --draft-season {args.draft_season}` first."
        )

    board = pd.read_csv(board_path)
    residuals = pd.read_csv(residuals_path)
    vet_residuals = residuals[residuals["is_rookie"] == 0]
    rookie_residuals = residuals[residuals["is_rookie"] == 1]

    board["_name_lower"] = board["player_display_name"].str.lower()
    matched_rows = []
    unmatched = []
    for name in names:
        rows = board[board["_name_lower"] == name.lower()]
        if rows.empty:
            unmatched.append(name)
            continue
        if len(rows) > 1:
            print(f"  warning: multiple board rows match '{name}' - using the first (highest-VBD)")
        matched_rows.append(rows.iloc[0])

    if unmatched:
        raise SystemExit(
            f"Could not find {len(unmatched)} player(s) on the board: {unmatched}\n"
            "Check spelling against the board's player_display_name column."
        )

    roster_board = pd.DataFrame(matched_rows).reset_index(drop=True)
    print(f"Simulating {len(roster_board)}-player roster ({args.n_sims:,} simulated seasons)...")
    for _, row in roster_board.iterrows():
        print(f"  {row['player_display_name']:25s} {row['position']:3s} ppg_pred={row['ppg_pred']:6.2f} "
              f"games_est={row['games_est']:5.2f} total_points_pred={row['total_points_pred']:7.2f}")

    result = simulate_roster_outcomes(roster_board, vet_residuals, rookie_residuals, n_sims=args.n_sims)

    print()
    print("=== Team season-total distribution ===")
    s = result["summary"]
    print(f"  p10:    {s['team_p10']:8.1f}")
    print(f"  p25:    {s['team_p25']:8.1f}")
    print(f"  median: {s['team_median']:8.1f}")
    print(f"  mean:   {s['team_mean']:8.1f}")
    print(f"  p75:    {s['team_p75']:8.1f}")
    print(f"  p90:    {s['team_p90']:8.1f}")

    print()
    print("=== Per-player contribution (sorted by mean simulated total) ===")
    print(result["player_detail"].round(1).to_string(index=False))


if __name__ == "__main__":
    main()

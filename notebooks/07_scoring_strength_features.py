"""
IPL Scoring Strength & Net Run Rate Features
==============================================
Adds pre-match features based on team scoring/bowling strength
and net run rate, all computed chronologically.

Batting:
- team1/team2_avg_runs_scored_last10, runs_scored_diff

Bowling:
- team1/team2_avg_runs_conceded_last10, runs_conceded_diff

Net Run Rate:
- team1/team2_nrr (cumulative expanding NRR)
- nrr_diff

Powerplay:
- team1/team2_powerplay_run_rate (last 10 matches avg)

Death Overs:
- team1/team2_death_over_run_rate (last 10 matches avg)

All calculations use ONLY prior matches (chronological, expanding).
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed" / "matches_with_momentum.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "matches_with_scoring.csv"


def normalise_team(name):
    TEAM_MAP = {
        "Delhi Daredevils": "Delhi Capitals",
        "Kings XI Punjab": "Punjab Kings",
        "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
        "Rising Pune Supergiant": "Rising Pune Supergiants",
        "Deccan Chargers": "Sunrisers Hyderabad",
    }
    return TEAM_MAP.get(name, name)


def build_scoring_features(df):
    """Build scoring strength and NRR features chronologically."""
    df = df.copy()
    df["batting_team"] = df["batting_team"].apply(normalise_team)
    df["bowling_team"] = df["bowling_team"].apply(normalise_team)
    df = df.sort_values("date").reset_index(drop=True)

    all_teams = sorted(
        set(df["batting_team"].unique()) | set(df["bowling_team"].unique())
    )

    # Track per-team stats
    team_stats = {}
    for t in all_teams:
        team_stats[t] = {
            "runs_scored": [],      # As batting team
            "runs_conceded": [],    # As bowling team (opposition's runs)
            "balls_faced": [],      # As batting team
            "balls_bowled": [],     # As bowling team (opposition's balls)
            "pp_runs_scored": [],   # Powerplay runs scored
            "pp_runs_conceded": [], # Powerplay runs conceded
            "death_runs_scored": [], # Death overs runs scored
            "death_runs_conceded": [], # Death overs runs conceded
            "pp_balls": [],         # PP balls faced/bowled
            "death_balls": [],      # Death overs balls faced/bowled
            # Cumulative for NRR
            "total_runs_scored": 0.0,
            "total_runs_conceded": 0.0,
            "total_balls_faced": 0.0,
            "total_balls_bowled": 0.0,
            "matches": 0,
        }

    # Output lists
    t1_runs_scored, t2_runs_scored = [], []
    t1_runs_conceded, t2_runs_conceded = [], []
    t1_nrr, t2_nrr = [], []
    t1_pp_rr, t2_pp_rr = [], []
    t1_death_rr, t2_death_rr = [], []

    for idx, row in df.iterrows():
        t1 = row["batting_team"]  # Team batting first
        t2 = row["bowling_team"]  # Team bowling first

        s1 = team_stats[t1]
        s2 = team_stats[t2]

        # --- Batting strength (last 10) ---
        t1_runs_scored.append(
            np.mean(s1["runs_scored"][-10:]) if s1["runs_scored"] else 160
        )
        t2_runs_scored.append(
            np.mean(s2["runs_scored"][-10:]) if s2["runs_scored"] else 160
        )

        # --- Bowling strength (last 10) ---
        t1_runs_conceded.append(
            np.mean(s1["runs_conceded"][-10:]) if s1["runs_conceded"] else 160
        )
        t2_runs_conceded.append(
            np.mean(s2["runs_conceded"][-10:]) if s2["runs_conceded"] else 160
        )

        # --- NRR (cumulative) ---
        t1_nrr.append(_calc_nrr(s1))
        t2_nrr.append(_calc_nrr(s2))

        # --- Powerplay run rate (last 10) ---
        t1_pp_rr.append(
            _safe_rr(s1["pp_runs_scored"][-10:], s1["pp_balls"][-10:])
            if s1["pp_runs_scored"] else 7.0
        )
        t2_pp_rr.append(
            _safe_rr(s2["pp_runs_scored"][-10:], s2["pp_balls"][-10:])
            if s2["pp_runs_scored"] else 7.0
        )

        # --- Death overs run rate (last 10) ---
        t1_death_rr.append(
            _safe_rr(s1["death_runs_scored"][-10:], s1["death_balls"][-10:])
            if s1["death_runs_scored"] else 8.0
        )
        t2_death_rr.append(
            _safe_rr(s2["death_runs_scored"][-10:], s2["death_balls"][-10:])
            if s2["death_runs_scored"] else 8.0
        )

        # --- Update stats AFTER computing features (no leakage) ---
        inn1_runs = row["inn1_runs"]
        inn2_runs = row["inn2_runs"]
        inn1_balls = row.get("inn1_balls_faced", 120)
        inn2_balls = row.get("inn2_balls_faced", 120)

        # Team 1 (batting first): scored inn1_runs, conceded inn2_runs
        s1["runs_scored"].append(inn1_runs)
        s1["runs_conceded"].append(inn2_runs)
        s1["balls_faced"].append(inn1_balls)
        s1["balls_bowled"].append(inn2_balls)
        s1["total_runs_scored"] += inn1_runs
        s1["total_runs_conceded"] += inn2_runs
        s1["total_balls_faced"] += inn1_balls
        s1["total_balls_bowled"] += inn2_balls
        s1["matches"] += 1
        s1["pp_runs_scored"].append(row.get("inn1_pp_runs", 0))
        s1["pp_runs_conceded"].append(row.get("inn2_pp_runs", 0))
        s1["death_runs_scored"].append(row.get("inn1_death_runs", 0))
        s1["death_runs_conceded"].append(row.get("inn2_death_runs", 0))
        s1["pp_balls"].append(36)  # 6 overs = 36 balls
        s1["death_balls"].append(
            max(0, inn1_balls - 96) if inn1_balls > 96 else 30
        )  # overs 16-20

        # Team 2 (bowling first): scored inn2_runs, conceded inn1_runs
        s2["runs_scored"].append(inn2_runs)
        s2["runs_conceded"].append(inn1_runs)
        s2["balls_faced"].append(inn2_balls)
        s2["balls_bowled"].append(inn1_balls)
        s2["total_runs_scored"] += inn2_runs
        s2["total_runs_conceded"] += inn1_runs
        s2["total_balls_faced"] += inn2_balls
        s2["total_balls_bowled"] += inn1_balls
        s2["matches"] += 1
        s2["pp_runs_scored"].append(row.get("inn2_pp_runs", 0))
        s2["pp_runs_conceded"].append(row.get("inn1_pp_runs", 0))
        s2["death_runs_scored"].append(row.get("inn2_death_runs", 0))
        s2["death_runs_conceded"].append(row.get("inn1_death_runs", 0))
        s2["pp_balls"].append(36)
        s2["death_balls"].append(
            max(0, inn2_balls - 96) if inn2_balls > 96 else 30
        )

    df["team1_avg_runs_scored_last10"] = t1_runs_scored
    df["team2_avg_runs_scored_last10"] = t2_runs_scored
    df["runs_scored_diff"] = np.array(t1_runs_scored) - np.array(t2_runs_scored)

    df["team1_avg_runs_conceded_last10"] = t1_runs_conceded
    df["team2_avg_runs_conceded_last10"] = t2_runs_conceded
    df["runs_conceded_diff"] = np.array(t1_runs_conceded) - np.array(t2_runs_conceded)

    df["team1_nrr"] = t1_nrr
    df["team2_nrr"] = t2_nrr
    df["nrr_diff"] = np.array(t1_nrr) - np.array(t2_nrr)

    df["team1_powerplay_run_rate"] = t1_pp_rr
    df["team2_powerplay_run_rate"] = t2_pp_rr

    df["team1_death_over_run_rate"] = t1_death_rr
    df["team2_death_over_run_rate"] = t2_death_rr

    return df


def _calc_nrr(stats):
    """Calculate NRR from cumulative stats. 0.0 for no matches."""
    if stats["matches"] == 0:
        return 0.0
    if stats["total_balls_faced"] == 0 or stats["total_balls_bowled"] == 0:
        return 0.0
    overs_faced = stats["total_balls_faced"] / 6
    overs_bowled = stats["total_balls_bowled"] / 6
    run_rate_for = stats["total_runs_scored"] / overs_faced
    run_rate_against = stats["total_runs_conceded"] / overs_bowled
    return run_rate_for - run_rate_against


def _safe_rr(runs_list, balls_list):
    """Safe run rate calculation for a list of values."""
    total_runs = sum(runs_list) if runs_list else 0
    total_balls = sum(balls_list) if balls_list else 1
    if total_balls == 0:
        return 0.0
    return total_runs / (total_balls / 6)


def print_scoring_summary(df):
    """Print scoring feature summary."""
    print("\nScoring Feature Summary:")

    features = [
        "runs_scored_diff", "runs_conceded_diff", "nrr_diff",
        "team1_nrr", "team2_nrr",
        "team1_avg_runs_scored_last10", "team2_avg_runs_scored_last10",
        "team1_avg_runs_conceded_last10", "team2_avg_runs_conceded_last10",
        "team1_powerplay_run_rate", "team2_powerplay_run_rate",
        "team1_death_over_run_rate", "team2_death_over_run_rate",
    ]
    for feat in features:
        if feat in df.columns:
            corr = df[feat].corr(df["team1_won"])
            print(f"  {feat:<40} r = {corr:+.4f}")

    # NRR distribution
    print(f"\n  NRR distribution (team1):")
    nrr = df["team1_nrr"]
    print(f"    Min: {nrr.min():.2f}, Max: {nrr.max():.2f}")
    print(f"    Mean: {nrr.mean():.2f}, Median: {nrr.median():.2f}")


def main():
    print("=" * 60)
    print("IPL Scoring Strength & NRR Features")
    print("=" * 60)

    print(f"\nLoading {INPUT_PATH} ...")
    if not Path(INPUT_PATH).exists():
        print("  Momentum file not found. Falling back...")
        venue_path = BASE_DIR / "data" / "processed" / "matches_with_venue.csv"
        if venue_path.exists():
            df = pd.read_csv(venue_path)
        else:
            df = pd.read_csv(BASE_DIR / "data" / "processed" / "matches.csv")
    else:
        df = pd.read_csv(INPUT_PATH)
    print(f"  {len(df)} matches")

    print("\nBuilding scoring strength features...")
    df = build_scoring_features(df)

    print_scoring_summary(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

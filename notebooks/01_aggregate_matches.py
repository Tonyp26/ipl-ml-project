"""
IPL Match-Level Data Aggregation
=================================
Aggregates ball-by-ball IPL data into match-level summaries
suitable for ML model training.

Features extracted per match:
- Teams, venue, toss decision & result
- Innings scores (runs, wickets, balls faced)
- Run rates (powerplay, middle, death overs)
- Partnership data
- Player of match
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_raw_data():
    """Load the ball-by-ball IPL dataset."""
    csv_path = DATA_DIR / "IPL.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"IPL.csv not found at {csv_path}")

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw shape: {df.shape}")
    return df


def aggregate_match_results(df):
    """Aggregate ball-by-ball data into one row per match."""

    # --- Match metadata (same for all rows in a match) ---
    match_meta = df.groupby("match_id").agg({
        "date": "first",
        "event_name": "first",
        "batting_team": "first",      # team batting first (innings 1)
        "bowling_team": "first",      # team bowling first
        "venue": "first",
        "city": "first",
        "day": "first",
        "month": "first",
        "year": "first",
        "season": "first",
        "gender": "first",
        "toss_winner": "first",
        "toss_decision": "first",
        "player_of_match": "first",
        "match_won_by": "first",
        "win_outcome": "first",
        "result_type": "first",
        "method": "first",
        "superover_winner": "first",
        "event_match_no": "first",
        "stage": "first",
    }).reset_index()

    # --- Innings-level aggregation ---
    # Innings 1 (first batting team)
    inn1 = df[df["innings"] == 1].groupby("match_id").agg({
        "runs_total": "sum",
        "team_wicket": ["first"],
        "team_balls": "max",
        "valid_ball": "sum",
    }).reset_index()
    inn1.columns = ["match_id", "inn1_runs", "inn1_wickets", "inn1_balls", "inn1_balls_faced"]

    # Innings 2 (second batting team)
    inn2 = df[df["innings"] == 2].groupby("match_id").agg({
        "runs_total": "sum",
        "team_wicket": ["first"],
        "team_balls": "max",
        "valid_ball": "sum",
    }).reset_index()
    inn2.columns = ["match_id", "inn2_runs", "inn2_wickets", "inn2_balls", "inn2_balls_faced"]

    # Powerplay runs (overs 1-6) for each innings
    pp1 = df[(df["innings"] == 1) & (df["over"] < 6)].groupby("match_id")["runs_total"].sum().reset_index()
    pp1.columns = ["match_id", "inn1_pp_runs"]
    pp2 = df[(df["innings"] == 2) & (df["over"] < 6)].groupby("match_id")["runs_total"].sum().reset_index()
    pp2.columns = ["match_id", "inn2_pp_runs"]

    # Death overs runs (overs 16-20) for each innings
    death1 = df[(df["innings"] == 1) & (df["over"] >= 16)].groupby("match_id")["runs_total"].sum().reset_index()
    death1.columns = ["match_id", "inn1_death_runs"]
    death2 = df[(df["innings"] == 2) & (df["over"] >= 16)].groupby("match_id")["runs_total"].sum().reset_index()
    death2.columns = ["match_id", "inn2_death_runs"]

    # Wickets in powerplay
    pp_wk1 = df[(df["innings"] == 1) & (df["over"] < 6) & (df["wicket_kind"].notna())].groupby("match_id").size().reset_index(name="inn1_pp_wickets")
    pp_wk2 = df[(df["innings"] == 2) & (df["over"] < 6) & (df["wicket_kind"].notna())].groupby("match_id").size().reset_index(name="inn2_pp_wickets")

    # Merge all innings features
    innings_features = inn1.merge(inn2, on="match_id", how="left")
    innings_features = innings_features.merge(pp1, on="match_id", how="left")
    innings_features = innings_features.merge(pp2, on="match_id", how="left")
    innings_features = innings_features.merge(death1, on="match_id", how="left")
    innings_features = innings_features.merge(death2, on="match_id", how="left")
    innings_features = innings_features.merge(pp_wk1, on="match_id", how="left")
    innings_features = innings_features.merge(pp_wk2, on="match_id", how="left")

    # Fill NaN for missing innings (some matches might have no innings 2 if rain/no result)
    innings_features = innings_features.fillna(0)

    # --- Calculate derived features ---
    # Run rates
    innings_features["inn1_run_rate"] = innings_features["inn1_runs"] / (innings_features["inn1_balls_faced"] / 6)
    innings_features["inn2_run_rate"] = innings_features["inn2_runs"] / (innings_features["inn2_balls_faced"] / 6)
    innings_features["inn1_pp_run_rate"] = innings_features["inn1_pp_runs"] / 6
    innings_features["inn2_pp_run_rate"] = innings_features["inn2_pp_runs"] / 6
    innings_features["inn1_death_run_rate"] = innings_features["inn1_death_runs"] / (
        (innings_features["inn1_balls_faced"] / 6 - 16).clip(lower=0.1)
    )
    innings_features["inn2_death_run_rate"] = innings_features["inn2_death_runs"] / (
        (innings_features["inn2_balls_faced"] / 6 - 16).clip(lower=0.1)
    )

    # Run rate difference
    innings_features["run_rate_diff"] = innings_features["inn2_run_rate"] - innings_features["inn1_run_rate"]

    # --- Merge metadata with innings features ---
    matches = match_meta.merge(innings_features, on="match_id", how="left")

    # --- Target variable: did team batting first win? ---
    matches["team1_won"] = (matches["match_won_by"] == matches["batting_team"]).astype(int)

    # --- Toss features ---
    matches["toss_winner_bat_first"] = (
        (matches["toss_winner"] == matches["batting_team"])
    ).astype(int)
    matches["toss_winner_field"] = (
        (matches["toss_decision"] == "field")
    ).astype(int)

    # --- Venue encoding ---
    venue_stats = df.groupby("venue")["runs_total"].mean()
    matches["venue_avg_score"] = matches["venue"].map(venue_stats)

    # --- Team frequency features (how often each team has played) ---
    team1_freq = df.groupby("batting_team")["match_id"].nunique()
    team2_freq = df.groupby("bowling_team")["match_id"].nunique()
    matches["team1_experience"] = matches["batting_team"].map(team1_freq)
    matches["team2_experience"] = matches["bowling_team"].map(team2_freq)

    print(f"  Aggregated matches: {len(matches)}")
    print(f"  Features: {len(matches.columns)}")

    return matches


def create_team_strength_features(matches, df):
    """
    Create rolling team strength features.
    For each match, compute each team's recent form (last 10 matches).
    """
    matches = matches.sort_values("date").reset_index(drop=True)

    # Build a rolling win record for each team
    all_teams = sorted(set(matches["batting_team"].unique()) | set(matches["bowling_team"].unique()))

    # Track cumulative wins and matches for each team
    team_wins = {t: 0 for t in all_teams}
    team_matches = {t: 0 for t in all_teams}

    # Rolling window features (last 10 matches)
    team_recent_wins = {t: [] for t in all_teams}

    records = []

    for idx, row in matches.iterrows():
        team1 = row["batting_team"]
        team2 = row["bowling_team"]
        date = row["date"]

        # Calculate recent form (last 10 matches before this one)
        prev_matches = matches[matches.index < idx]

        # Team 1 recent wins
        t1_recent = prev_matches[
            ((prev_matches["batting_team"] == team1) & (prev_matches["team1_won"] == 1)) |
            ((prev_matches["bowling_team"] == team1) & (prev_matches["team1_won"] == 0))
        ].tail(10)
        team1_recent_win_pct = len(t1_recent) / 10 if len(t1_recent) > 0 else 0.5

        # Team 2 recent wins
        t2_recent = prev_matches[
            ((prev_matches["batting_team"] == team2) & (prev_matches["team1_won"] == 1)) |
            ((prev_matches["bowling_team"] == team2) & (prev_matches["team1_won"] == 0))
        ].tail(10)
        team2_recent_win_pct = len(t2_recent) / 10 if len(t2_recent) > 0 else 0.5

        # Cumulative win rates
        team_matches[team1] += 1
        team_matches[team2] += 1
        if row["team1_won"] == 1:
            team_wins[team1] += 1
        else:
            team_wins[team2] += 1

        records.append({
            "match_id": row["match_id"],
            "team1_recent_win_pct": team1_recent_win_pct,
            "team2_recent_win_pct": team2_recent_win_pct,
            "team1_cumulative_win_pct": team_wins[team1] / team_matches[team1],
            "team2_cumulative_win_pct": team_wins[team2] / team_matches[team2],
            "team1_total_matches": team_matches[team1],
            "team2_total_matches": team_matches[team2],
        })

    form_df = pd.DataFrame(records)
    matches = matches.merge(form_df, on="match_id", how="left")

    return matches


def save_processed_data(matches):
    """Save processed match-level data."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "matches.csv"
    matches.to_csv(output_path, index=False)
    print(f"\nSaved processed matches to {output_path}")
    print(f"  Shape: {matches.shape}")
    return output_path


def main():
    print("=" * 60)
    print("IPL Data Aggregation Pipeline")
    print("=" * 60)

    # Step 1: Load raw data
    print("\n[1/3] Loading raw ball-by-ball data...")
    df = load_raw_data()

    # Step 2: Aggregate to match level
    print("\n[2/3] Aggregating to match-level summaries...")
    matches = aggregate_match_results(df)

    # Step 3: Add team strength features
    print("\n[3/3] Computing team strength features...")
    matches = create_team_strength_features(matches, df)

    # Save
    output_path = save_processed_data(matches)

    # Show summary
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Total matches: {len(matches)}")
    print(f"Seasons covered: {matches['season'].nunique()} ({matches['season'].min()} - {matches['season'].max()})")
    print(f"Unique venues: {matches['venue'].nunique()}")
    print(f"Teams: {sorted(set(matches['batting_team'].unique()) | set(matches['bowling_team'].unique()))}")
    print(f"Team 1 (batting first) win rate: {matches['team1_won'].mean():.1%}")
    print(f"Toss winner choosing field: {matches['toss_winner_field'].mean():.1%}")

    return matches


if __name__ == "__main__":
    main()

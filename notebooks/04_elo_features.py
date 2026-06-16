"""
IPL Elo Rating Features
========================
Calculates chronological Elo ratings for each IPL team.
- All teams start at 1500
- K=32 (standard for sports)
- Home advantage: +65 Elo points for home team
- Ratings updated AFTER each match (no leakage)
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "matches.csv"
ELO_OUTPUT = BASE_DIR / "data" / "processed" / "matches_with_elo.csv"

K = 32
HOME_ADVANTAGE = 65  # Equivalent to ~1.6x expected score boost


def normalise_team(name):
    TEAM_MAP = {
        "Delhi Daredevils": "Delhi Capitals",
        "Kings XI Punjab": "Punjab Kings",
        "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
        "Rising Pune Supergiant": "Rising Pune Supergiants",
        "Deccan Chargers": "Sunrisers Hyderabad",
    }
    return TEAM_MAP.get(name, name)


def calculate_elo_ratings(df):
    """
    Compute Elo ratings chronologically.
    For each match, returns the Elo rating EACH TEAM had BEFORE the match.
    """
    df = df.copy()
    df["batting_team"] = df["batting_team"].apply(normalise_team)
    df["bowling_team"] = df["bowling_team"].apply(normalise_team)
    df = df.sort_values("date").reset_index(drop=True)

    all_teams = sorted(
        set(df["batting_team"].unique()) | set(df["bowling_team"].unique())
    )
    elo = {team: 1500.0 for team in all_teams}

    team1_elo, team2_elo = [], []

    for idx, row in df.iterrows():
        t1 = row["batting_team"]
        t2 = row["bowling_team"]

        # Record pre-match Elo
        team1_elo.append(elo[t1])
        team2_elo.append(elo[t2])

        # Compute expected score with home advantage
        t1_elo_adj = elo[t1] + HOME_ADVANTAGE if row["batting_team"] == row["city"] else elo[t1]
        t2_elo_adj = elo[t2] + HOME_ADVANTAGE if row["bowling_team"] == row["city"] else elo[t2]

        expected_t1 = 1 / (1 + 10 ** ((t2_elo_adj - t1_elo_adj) / 400))
        expected_t2 = 1 - expected_t1

        # Actual outcome (1 = team1/batting first won)
        actual_t1 = int(row["team1_won"])
        actual_t2 = 1 - actual_t1

        # Update Elo
        elo[t1] += K * (actual_t1 - expected_t1)
        elo[t2] += K * (actual_t2 - expected_t2)

    df["team1_elo"] = team1_elo
    df["team2_elo"] = team2_elo
    df["elo_diff"] = df["team1_elo"] - df["team2_elo"]

    return df


def print_elo_summary(df):
    """Print Elo rating summary."""
    print("\nElo Rating Summary:")
    print(f"  K-factor: {K}")
    print(f"  Home advantage: +{HOME_ADVANTAGE}")

    # Final Elo ratings
    final_elo = {}
    for idx, row in df.iloc[::-1].iterrows():
        t1, t2 = row["batting_team"], row["bowling_team"]
        if t1 not in final_elo:
            final_elo[t1] = row["team1_elo"]
        if t2 not in final_elo:
            final_elo[t2] = row["team2_elo"]
        if len(final_elo) >= len(set(df["batting_team"]) | set(df["bowling_team"])):
            break

    print("\n  Final Elo ratings (last appearance):")
    for team, rating in sorted(final_elo.items(), key=lambda x: x[1], reverse=True):
        delta = rating - 1500
        sign = "+" if delta >= 0 else ""
        print(f"    {team:<35} {rating:>7.1f}  ({sign}{delta:.0f})")

    # Elo diff correlation with outcome
    corr = df["elo_diff"].corr(df["team1_won"])
    print(f"\n  Elo diff correlation with outcome: {corr:.4f}")


def main():
    print("=" * 60)
    print("IPL Elo Rating Feature Generation")
    print("=" * 60)

    print(f"\nLoading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} matches")

    print("\nCalculating Elo ratings...")
    df = calculate_elo_ratings(df)

    print_elo_summary(df)

    ELO_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ELO_OUTPUT, index=False)
    print(f"\nSaved: {ELO_OUTPUT}")


if __name__ == "__main__":
    main()

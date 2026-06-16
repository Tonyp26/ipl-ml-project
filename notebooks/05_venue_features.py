"""
IPL Venue-Specific Team Strength Features
===========================================
Adds pre-match features based on each team's historical performance
at specific venues.

Features added per match:
- team1_venue_win_pct / team2_venue_win_pct
- venue_win_pct_diff
- team1_venue_matches / team2_venue_matches
- team1_avg_runs_at_venue / team2_avg_runs_at_venue

All calculations use ONLY past matches (chronological, expanding).
Minimum sample threshold: 3 matches. Falls back to global team stats.
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed" / "matches_with_elo.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "matches_with_venue.csv"

MIN_VENUE_MATCHES = 3  # Minimum matches before falling back to global


def normalise_team(name):
    TEAM_MAP = {
        "Delhi Daredevils": "Delhi Capitals",
        "Kings XI Punjab": "Punjab Kings",
        "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
        "Rising Pune Supergiant": "Rising Pune Supergiants",
        "Deccan Chargers": "Sunrisers Hyderabad",
    }
    return TEAM_MAP.get(name, name)


def build_venue_features(df):
    """
    Build venue-specific team strength features chronologically.
    For each match, compute each team's historical performance at that venue
    using only PRIOR matches.
    """
    df = df.copy()
    df["batting_team"] = df["batting_team"].apply(normalise_team)
    df["bowling_team"] = df["bowling_team"].apply(normalise_team)
    df = df.sort_values("date").reset_index(drop=True)

    all_teams = sorted(
        set(df["batting_team"].unique()) | set(df["bowling_team"].unique())
    )

    # Track per team per venue
    # venue_stats[team][venue] = {"wins": n, "matches": n, "runs": [], "global_wins": n, "global_matches": n}
    venue_stats = {}
    for team in all_teams:
        venue_stats[team] = {}

    # Global team stats (fallback)
    global_stats = {
        team: {"wins": 0, "matches": 0, "runs": []}
        for team in all_teams
    }

    t1_venue_pct, t2_venue_pct = [], []
    t1_venue_matches, t2_venue_matches = [], []
    t1_avg_runs, t2_avg_runs = [], []

    for idx, row in df.iterrows():
        t1 = row["batting_team"]
        t2 = row["bowling_team"]
        venue = row["venue"]

        # Team 1 venue stats
        v1 = venue_stats[t1].get(venue, {"wins": 0, "matches": 0, "runs": []})
        if v1["matches"] >= MIN_VENUE_MATCHES:
            t1_venue_pct.append(v1["wins"] / v1["matches"])
            t1_avg_runs.append(np.mean(v1["runs"]) if v1["runs"] else 160)
        else:
            # Fallback to global team stats
            g1 = global_stats[t1]
            if g1["matches"] > 0:
                t1_venue_pct.append(g1["wins"] / g1["matches"])
                t1_avg_runs.append(np.mean(g1["runs"]) if g1["runs"] else 160)
            else:
                t1_venue_pct.append(0.5)
                t1_avg_runs.append(160)
        t1_venue_matches.append(v1["matches"])

        # Team 2 venue stats
        v2 = venue_stats[t2].get(venue, {"wins": 0, "matches": 0, "runs": []})
        if v2["matches"] >= MIN_VENUE_MATCHES:
            t2_venue_pct.append(v2["wins"] / v2["matches"])
            t2_avg_runs.append(np.mean(v2["runs"]) if v2["runs"] else 160)
        else:
            g2 = global_stats[t2]
            if g2["matches"] > 0:
                t2_venue_pct.append(g2["wins"] / g2["matches"])
                t2_avg_runs.append(np.mean(g2["runs"]) if g2["runs"] else 160)
            else:
                t2_venue_pct.append(0.5)
                t2_avg_runs.append(160)
        t2_venue_matches.append(v2["matches"])

        # Update stats AFTER computing features (no leakage)
        won = int(row["team1_won"])

        # Update team 1 stats
        if venue not in venue_stats[t1]:
            venue_stats[t1][venue] = {"wins": 0, "matches": 0, "runs": []}
        venue_stats[t1][venue]["matches"] += 1
        venue_stats[t1][venue]["wins"] += won
        venue_stats[t1][venue]["runs"].append(row["inn1_runs"])

        global_stats[t1]["matches"] += 1
        global_stats[t1]["wins"] += won
        global_stats[t1]["runs"].append(row["inn1_runs"])

        # Update team 2 stats (team2 won when team1_won=0)
        t2_won = 1 - won
        if venue not in venue_stats[t2]:
            venue_stats[t2][venue] = {"wins": 0, "matches": 0, "runs": []}
        venue_stats[t2][venue]["matches"] += 1
        venue_stats[t2][venue]["wins"] += t2_won
        venue_stats[t2][venue]["runs"].append(row["inn2_runs"])

        global_stats[t2]["matches"] += 1
        global_stats[t2]["wins"] += t2_won
        global_stats[t2]["runs"].append(row["inn2_runs"])

    df["team1_venue_win_pct"] = t1_venue_pct
    df["team2_venue_win_pct"] = t2_venue_pct
    df["venue_win_pct_diff"] = np.array(t1_venue_pct) - np.array(t2_venue_pct)
    df["team1_venue_matches"] = t1_venue_matches
    df["team2_venue_matches"] = t2_venue_matches
    df["team1_avg_runs_at_venue"] = t1_avg_runs
    df["team2_avg_runs_at_venue"] = t2_avg_runs

    return df


def print_venue_summary(df):
    """Print venue feature summary."""
    print("\nVenue Feature Summary:")

    # Distribution of venue-specific vs fallback
    t1_venue_specific = (df["team1_venue_matches"] >= MIN_VENUE_MATCHES).sum()
    t2_venue_specific = (df["team2_venue_matches"] >= MIN_VENUE_MATCHES).sum()
    total = len(df)
    print(f"  Matches with venue-specific data:")
    print(f"    Team 1: {t1_venue_specific}/{total} ({t1_venue_specific/total:.1%})")
    print(f"    Team 2: {t2_venue_specific}/{total} ({t2_venue_specific/total:.1%})")

    # Correlations with outcome
    for feat in ["venue_win_pct_diff", "team1_venue_win_pct", "team2_venue_win_pct",
                 "team1_avg_runs_at_venue", "team2_avg_runs_at_venue"]:
        if feat in df.columns:
            corr = df[feat].corr(df["team1_won"])
            print(f"  {feat:<30} r = {corr:+.4f}")

    # Top venues by matches
    print(f"\n  Top venues (by match count):")
    venue_counts = df["venue"].value_counts().head(10)
    for venue, count in venue_counts.items():
        short = venue[:40]
        print(f"    {short:<40} {count} matches")


def main():
    print("=" * 60)
    print("IPL Venue-Specific Team Strength Features")
    print("=" * 60)

    print(f"\nLoading {INPUT_PATH} ...")
    if not Path(INPUT_PATH).exists():
        print("  Elo file not found. Loading base matches.csv...")
        df = pd.read_csv(BASE_DIR / "data" / "processed" / "matches.csv")
    else:
        df = pd.read_csv(INPUT_PATH)
    print(f"  {len(df)} matches")

    print("\nBuilding venue-specific features...")
    df = build_venue_features(df)

    print_venue_summary(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

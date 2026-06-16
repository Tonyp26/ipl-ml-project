"""
IPL Momentum Features
======================
Adds pre-match momentum features based on each team's recent form.

Features added per match:
- team1/team2_last3_win_pct, last3_diff
- team1/team2_last5_win_pct, last5_diff
- team1/team2_current_win_streak, streak_diff
- team1/team2_matches_since_loss

All calculations use ONLY past matches (chronological, expanding).
"""

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed" / "matches_with_venue.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "matches_with_momentum.csv"


def normalise_team(name):
    TEAM_MAP = {
        "Delhi Daredevils": "Delhi Capitals",
        "Kings XI Punjab": "Punjab Kings",
        "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
        "Rising Pune Supergiant": "Rising Pune Supergiants",
        "Deccan Chargers": "Sunrisers Hyderabad",
    }
    return TEAM_MAP.get(name, name)


def build_momentum_features(df):
    """
    Build momentum features chronologically.
    For each match, compute each team's recent form using ONLY prior matches.
    """
    df = df.copy()
    df["batting_team"] = df["batting_team"].apply(normalise_team)
    df["bowling_team"] = df["bowling_team"].apply(normalise_team)
    df = df.sort_values("date").reset_index(drop=True)

    all_teams = sorted(
        set(df["batting_team"].unique()) | set(df["bowling_team"].unique())
    )

    # Track each team's match history (results list: 1=win, 0=loss)
    team_history = {team: [] for team in all_teams}

    t1_last3, t2_last3 = [], []
    t1_last5, t2_last5 = [], []
    t1_streak, t2_streak = [], []
    t1_since_loss, t2_since_loss = [], []

    for idx, row in df.iterrows():
        t1 = row["batting_team"]
        t2 = row["bowling_team"]

        h1 = team_history[t1]
        h2 = team_history[t2]

        # Last 3 win %
        t1_last3.append(np.mean(h1[-3:]) if len(h1) >= 3 else 0.5)
        t2_last3.append(np.mean(h2[-3:]) if len(h2) >= 3 else 0.5)

        # Last 5 win %
        t1_last5.append(np.mean(h1[-5:]) if len(h1) >= 5 else 0.5)
        t2_last5.append(np.mean(h2[-5:]) if len(h2) >= 5 else 0.5)

        # Current win streak (consecutive wins from most recent match)
        t1_streak.append(_current_streak(h1))
        t2_streak.append(_current_streak(h2))

        # Matches since last loss
        t1_since_loss.append(_matches_since_loss(h1))
        t2_since_loss.append(_matches_since_loss(h2))

        # Update history AFTER computing features (no leakage)
        won = int(row["team1_won"])
        team_history[t1].append(won)
        team_history[t2].append(1 - won)

    df["team1_last3_win_pct"] = t1_last3
    df["team2_last3_win_pct"] = t2_last3
    df["last3_diff"] = np.array(t1_last3) - np.array(t2_last3)

    df["team1_last5_win_pct"] = t1_last5
    df["team2_last5_win_pct"] = t2_last5
    df["last5_diff"] = np.array(t1_last5) - np.array(t2_last5)

    df["team1_current_win_streak"] = t1_streak
    df["team2_current_win_streak"] = t2_streak
    df["streak_diff"] = np.array(t1_streak) - np.array(t2_streak)

    df["team1_matches_since_loss"] = t1_since_loss
    df["team2_matches_since_loss"] = t2_since_loss

    return df


def _current_streak(history):
    """Count consecutive wins from the end of the history."""
    if not history:
        return 0
    streak = 0
    for result in reversed(history):
        if result == 1:
            streak += 1
        else:
            break
    return streak


def _matches_since_loss(history):
    """Count matches since the last loss."""
    if not history:
        return 0
    count = 0
    for result in reversed(history):
        if result == 0:
            break
        count += 1
    return count


def print_momentum_summary(df):
    """Print momentum feature summary."""
    print("\nMomentum Feature Summary:")

    # Correlations with outcome
    features = [
        "last3_diff", "last5_diff", "streak_diff",
        "team1_last3_win_pct", "team2_last3_win_pct",
        "team1_last5_win_pct", "team2_last5_win_pct",
        "team1_current_win_streak", "team2_current_win_streak",
        "team1_matches_since_loss", "team2_matches_since_loss",
    ]
    for feat in features:
        if feat in df.columns:
            corr = df[feat].corr(df["team1_won"])
            print(f"  {feat:<35} r = {corr:+.4f}")

    # Distribution stats
    print(f"\n  Win streak distribution:")
    streaks = df["team1_current_win_streak"]
    print(f"    Max: {streaks.max()}, Mean: {streaks.mean():.1f}, "
          f"Median: {streaks.median():.0f}")
    print(f"    3+ streak: {(streaks >= 3).sum()} matches ({(streaks >= 3).mean():.1%})")
    print(f"    5+ streak: {(streaks >= 5).sum()} matches ({(streaks >= 5).mean():.1%})")

    print(f"\n  Matches since loss distribution:")
    since = df["team1_matches_since_loss"]
    print(f"    Max: {since.max()}, Mean: {since.mean():.1f}, "
          f"Median: {since.median():.0f}")
    print(f"    3+ since loss: {(since >= 3).sum()} matches ({(since >= 3).mean():.1%})")


def main():
    print("=" * 60)
    print("IPL Momentum Features")
    print("=" * 60)

    print(f"\nLoading {INPUT_PATH} ...")
    if not Path(INPUT_PATH).exists():
        print("  Venue file not found. Falling back...")
        elo_path = BASE_DIR / "data" / "processed" / "matches_with_elo.csv"
        if elo_path.exists():
            df = pd.read_csv(elo_path)
        else:
            df = pd.read_csv(BASE_DIR / "data" / "processed" / "matches.csv")
    else:
        df = pd.read_csv(INPUT_PATH)
    print(f"  {len(df)} matches")

    print("\nBuilding momentum features...")
    df = build_momentum_features(df)

    print_momentum_summary(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

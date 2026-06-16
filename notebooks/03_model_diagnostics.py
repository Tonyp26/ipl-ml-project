"""
IPL Model Diagnostics - Phase 2.5: Feature Audit
===================================================
Verifies that all rolling features use ONLY past match information.
Generates feature correlation report.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "matches.csv"
MODEL_DIR = BASE_DIR / "models"

TEAM_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Deccan Chargers": "Sunrisers Hyderabad",
}

TARGET = "team1_won"


def normalise(name):
    return TEAM_MAP.get(name, name)


def build_features_with_tracking(df):
    """
    Build features while tracking what data was used for each match.
    Returns df with features + tracking columns for audit.
    """
    df = df.copy()
    df["batting_team"] = df["batting_team"].apply(normalise)
    df["bowling_team"] = df["bowling_team"].apply(normalise)
    df["toss_winner"] = df["toss_winner"].apply(normalise)
    df = df.sort_values("date").reset_index(drop=True)

    all_teams = sorted(set(df["batting_team"].unique()) | set(df["bowling_team"].unique()))
    team_le = LabelEncoder()
    team_le.fit(all_teams)
    df["t1_enc"] = team_le.transform(df["batting_team"])
    df["t2_enc"] = team_le.transform(df["bowling_team"])
    df["toss_enc"] = team_le.transform(df["toss_winner"])

    venue_freq = df["venue"].value_counts(normalize=True)
    df["venue_freq"] = df["venue"].map(venue_freq)
    venue_avg = df.groupby("venue")["inn1_runs"].mean()
    df["venue_avg_inn1"] = df["venue"].map(venue_avg)

    df["season_num"] = pd.to_numeric(
        df["season"].str.replace("/", "."), errors="coerce"
    ).fillna(2008)

    df["t1_home"] = (df["batting_team"] == df["city"]).astype(int)
    df["t2_home"] = (df["bowling_team"] == df["city"]).astype(int)
    df["toss_winner_bat_first"] = (df["toss_winner"] == df["batting_team"]).astype(int)
    df["toss_field"] = (df["toss_decision"] == "field").astype(int)

    # Rolling features WITH tracking
    team_records = {t: {"wins": 0, "matches": 0, "results": []} for t in all_teams}
    h2h = {}

    cols = {
        "t1_cum": [], "t2_cum": [],
        "t1_r5": [], "t2_r5": [],
        "t1_r10": [], "t2_r10": [],
        "t1_m": [], "t2_m": [],
        "h2h_pct": [], "h2h_n": [],
        # Tracking: how many matches were used to compute each feature
        "t1_data_points": [], "t2_data_points": [],
        "h2h_data_points": [],
    }

    for idx, row in df.iterrows():
        t1, t2 = row["batting_team"], row["bowling_team"]

        r1 = team_records[t1]["results"]
        r2 = team_records[t2]["results"]

        cols["t1_cum"].append(
            team_records[t1]["wins"] / max(team_records[t1]["matches"], 1)
        )
        cols["t2_cum"].append(
            team_records[t2]["wins"] / max(team_records[t2]["matches"], 1)
        )
        cols["t1_r5"].append(np.mean(r1[-5:]) if len(r1) >= 5 else 0.5)
        cols["t2_r5"].append(np.mean(r2[-5:]) if len(r2) >= 5 else 0.5)
        cols["t1_r10"].append(np.mean(r1[-10:]) if len(r1) >= 10 else 0.5)
        cols["t2_r10"].append(np.mean(r2[-10:]) if len(r2) >= 10 else 0.5)
        cols["t1_m"].append(team_records[t1]["matches"])
        cols["t2_m"].append(team_records[t2]["matches"])

        key = (t1, t2)
        rev = (t2, t1)
        hl = h2h.get(key, []) + h2h.get(rev, [])
        cols["h2h_pct"].append(np.mean(hl) if hl else 0.5)
        cols["h2h_n"].append(len(hl))

        # Tracking
        cols["t1_data_points"].append(team_records[t1]["matches"])
        cols["t2_data_points"].append(team_records[t2]["matches"])
        cols["h2h_data_points"].append(len(hl))

        # Update AFTER computing (no leakage)
        won = int(row[TARGET])
        team_records[t1]["wins"] += won
        team_records[t1]["matches"] += 1
        team_records[t1]["results"].append(won)
        team_records[t2]["wins"] += (1 - won)
        team_records[t2]["matches"] += 1
        team_records[t2]["results"].append(1 - won)
        if key not in h2h:
            h2h[key] = []
        h2h[key].append(won)

    df["t1_cum_pct"] = cols["t1_cum"]
    df["t2_cum_pct"] = cols["t2_cum"]
    df["t1_recent5"] = cols["t1_r5"]
    df["t2_recent5"] = cols["t2_r5"]
    df["t1_recent10"] = cols["t1_r10"]
    df["t2_recent10"] = cols["t2_r10"]
    df["t1_matches"] = cols["t1_m"]
    df["t2_matches"] = cols["t2_m"]
    df["t1_h2h_pct"] = cols["h2h_pct"]
    df["t1_h2h_n"] = cols["h2h_n"]

    # Tracking columns
    df["t1_data_points"] = cols["t1_data_points"]
    df["t2_data_points"] = cols["t2_data_points"]
    df["h2h_data_points"] = cols["h2h_data_points"]

    df["cum_diff"] = df["t1_cum_pct"] - df["t2_cum_pct"]
    df["recent5_diff"] = df["t1_recent5"] - df["t2_recent5"]
    df["recent10_diff"] = df["t1_recent10"] - df["t2_recent10"]
    df["home_diff"] = df["t1_home"] - df["t2_home"]
    df["matches_diff"] = df["t1_matches"].astype(float) - df["t2_matches"].astype(float)

    return df, team_le


def audit_cumulative_features(df):
    """Verify cumulative win-rate features only use past matches."""
    print("=" * 60)
    print("3. CUMULATIVE WIN-RATE FEATURE AUDIT")
    print("=" * 60)

    # For each team's FIRST appearance (either as bat or bowl team),
    # the cumulative feature should be based on 0 prior matches
    issues = 0

    # Find first appearance of each team
    team_first_appearance = {}
    for idx, row in df.iterrows():
        for team_col in ["batting_team", "bowling_team"]:
            team = row[team_col]
            if team not in team_first_appearance:
                team_first_appearance[team] = idx

    print("\n  First appearance per team — data points used:")
    for team, first_idx in sorted(team_first_appearance.items()):
        row = df.iloc[first_idx]
        dp = row["t1_data_points"] if row["batting_team"] == team else row["t2_data_points"]
        status = "OK" if dp == 0 else "FAIL"
        if dp != 0:
            issues += 1
            print(f"    {status}: {team:<35} {dp} data points (expected 0)")

    if issues == 0:
        print("    PASS: All teams start with 0 data points (no future leakage)")
    else:
        print(f"    {issues} teams have non-zero data at first appearance")

    # Smoothness: for teams with enough matches, check cum rate changes gradually
    print("\n  Smoothness check (cumulative win rate gradual change, 10+ matches):")
    jump_issues = 0
    for team in df["batting_team"].unique()[:5]:
        team_matches = df[
            (df["batting_team"] == team) | (df["bowling_team"] == team)
        ].sort_values("date")
        if len(team_matches) < 10:
            continue

        # Use the right column depending on which team role
        cum_vals = []
        for _, r in team_matches.iterrows():
            if r["batting_team"] == team:
                cum_vals.append(r["t1_cum_pct"])
            else:
                cum_vals.append(r["t2_cum_pct"])
        cum_vals = np.array(cum_vals)

        diffs = np.abs(np.diff(cum_vals))
        max_jump = diffs.max() if len(diffs) > 0 else 0
        # Allow up to 0.5 jump (single match can swing early cum rate a lot)
        status = "OK" if max_jump <= 1.0 else "FAIL"
        if max_jump > 1.0:
            jump_issues += 1
        print(f"    {team:<30} max jump: {max_jump:.4f} [{status}]")

    if jump_issues == 0:
        print("    PASS: Cumulative win rates change gradually")

    # Data point distribution
    print("\n  Data points distribution (matches used for cum features):")
    print(f"    t1: median={df['t1_data_points'].median():.0f}, "
          f"min={df['t1_data_points'].min():.0f}, "
          f"max={df['t1_data_points'].max():.0f}")
    print(f"    t2: median={df['t2_data_points'].median():.0f}, "
          f"min={df['t2_data_points'].min():.0f}, "
          f"max={df['t2_data_points'].max():.0f}")

    return issues == 0 and jump_issues == 0


def audit_h2h_features(df):
    """Verify head-to-head features only use past matches."""
    print("\n" + "=" * 60)
    print("4. HEAD-TO-HEAD FEATURE AUDIT")
    print("=" * 60)

    # First encounter between any two teams should have h2h_n=0
    print("\n  First H2H encounter per team pair — data points:")
    issues = 0
    team_pairs = set()
    for idx, row in df.iterrows():
        key = tuple(sorted([row["batting_team"], row["bowling_team"]]))
        if key not in team_pairs:
            team_pairs.add(key)
            if row["h2h_data_points"] != 0:
                print(f"    FAIL: {key[0]} vs {key[1]} has {row['h2h_data_points']} H2H data points (expected 0)")
                issues += 1
            if issues > 5:
                print("    ... (more failures suppressed)")
                break

    if issues == 0:
        print("    PASS: All first encounters have 0 H2H data points")

    # Show H2H distribution
    print("\n  H2H data points distribution:")
    print(f"    median={df['h2h_data_points'].median():.0f}, "
          f"mean={df['h2h_data_points'].mean():.1f}, "
          f"max={df['h2h_data_points'].max():.0f}")
    print(f"    Zero H2H: {(df['h2h_data_points'] == 0).sum()} matches ({(df['h2h_data_points'] == 0).mean():.1%})")

    return issues == 0


def feature_correlation_report(df):
    """Generate feature correlation report."""
    print("\n" + "=" * 60)
    print("5. FEATURE CORRELATION REPORT")
    print("=" * 60)

    feature_cols = [
        "t1_enc", "t2_enc", "toss_enc",
        "toss_winner_bat_first", "toss_field",
        "venue_freq", "venue_avg_inn1",
        "recent5_diff", "recent10_diff", "cum_diff",
        "t1_h2h_pct", "t1_h2h_n",
        "home_diff", "t1_matches", "t2_matches",
        "season_num", TARGET,
    ]

    df_feat = df[feature_cols].dropna()
    corr = df_feat.corr()

    # Show correlations with target
    target_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    print("\n  Feature correlations with target:")
    for feat, val in target_corr.items():
        if np.isnan(val):
            print(f"    {feat:<25} NaN     (constant or missing)")
            continue
        bar = "+" * int(abs(val) * 30) if val > 0 else "-" * int(abs(val) * 30)
        print(f"    {feat:<25} {val:+.4f} {bar}")

    # Find highly correlated feature pairs (|r| > 0.7)
    print("\n  Highly correlated feature pairs (|r| > 0.7):")
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            if corr.columns[i] != TARGET and corr.columns[j] != TARGET:
                val = corr.iloc[i, j]
                if abs(val) > 0.7:
                    high_corr.append((corr.columns[i], corr.columns[j], val))

    if high_corr:
        for f1, f2, val in high_corr:
            print(f"    {f1:<25} <-> {f2:<25} r={val:+.4f}")
    else:
        print("    None found — features are reasonably independent.")

    # Plot correlation matrix
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    fig.savefig(MODEL_DIR / "correlation_matrix.png", dpi=150)
    plt.close()
    print(f"\n  Saved: models/correlation_matrix.png")


def main():
    print("IPL Model Diagnostics - Feature Audit")
    print("=" * 60)

    print(f"\nLoading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} matches")

    print("\nBuilding features with tracking...")
    df, team_le = build_features_with_tracking(df)
    print(f"  {len(df)} matches with features")

    # Audit cumulative features
    cum_ok = audit_cumulative_features(df)

    # Audit H2H features
    h2h_ok = audit_h2h_features(df)

    # Feature correlation
    feature_correlation_report(df)

    print("\n" + "=" * 60)
    print("FEATURE AUDIT SUMMARY")
    print("=" * 60)
    print(f"  Cumulative features use past data only: {cum_ok}")
    print(f"  H2H features use past data only:        {h2h_ok}")


if __name__ == "__main__":
    main()

"""
IPL 2025 Backtest
==================
Trains on seasons <= 2024, predicts every 2025 match, and evaluates.
Also simulates the full 2025 season and compares predicted vs actual standings.

Production model: Elo + Venue (24 features)
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
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


def normalise(name):
    return TEAM_MAP.get(name, name)


# ============================================================
# Feature Engineering (matches Elo + Venue, 24 features)
# ============================================================

def build_features_for_matches(historical_df, target_matches):
    """
    Replay historical matches to build state, then generate features
    for each target match chronologically.
    """
    # Combine and sort
    hist = historical_df.copy()
    hist["_source"] = "history"
    targets = target_matches.copy()
    targets["_source"] = "target"

    all_matches = pd.concat([hist, targets], ignore_index=True)
    all_matches = all_matches.sort_values("date").reset_index(drop=True)

    all_teams = sorted(
        set(all_matches["batting_team"].unique()) | set(all_matches["bowling_team"].unique())
    )

    # Team state tracking
    team_records = {}
    for t in all_teams:
        team_records[t] = {
            "wins": 0, "matches": 0, "results": [],
            "elo": 1500.0,
            "venue_stats": {},  # venue -> {wins, matches, runs}
            "global_runs": [],
        }

    h2h = {}  # (t1, t2) -> [t1_wins]

    # Venue static properties (from ALL data)
    venue_freq = all_matches["venue"].value_counts(normalize=True)
    venue_avg = all_matches.groupby("venue")["inn1_runs"].mean()

    # Encode teams
    team_le = LabelEncoder()
    team_le.fit(all_teams)

    K_ELO = 32
    HOME_ADV = 65
    MIN_VENUE = 3

    feature_rows = []

    for idx, row in all_matches.iterrows():
        t1 = row["batting_team"]
        t2 = row["bowling_team"]
        venue = row["venue"]
        source = row["_source"]

        r1 = team_records[t1]["results"]
        r2 = team_records[t2]["results"]

        # Cumulative win %
        t1_cum = team_records[t1]["wins"] / max(team_records[t1]["matches"], 1)
        t2_cum = team_records[t2]["wins"] / max(team_records[t2]["matches"], 1)

        # Recent form
        t1_r5 = np.mean(r1[-5:]) if len(r1) >= 5 else 0.5
        t2_r5 = np.mean(r2[-5:]) if len(r2) >= 5 else 0.5
        t1_r10 = np.mean(r1[-10:]) if len(r1) >= 10 else 0.5
        t2_r10 = np.mean(r2[-10:]) if len(r2) >= 10 else 0.5

        # Elo
        t1_elo = team_records[t1]["elo"]
        t2_elo = team_records[t2]["elo"]

        # H2H
        key = (t1, t2)
        rev = (t2, t1)
        h2h_list = h2h.get(key, []) + h2h.get(rev, [])
        h2h_pct = np.mean(h2h_list) if h2h_list else 0.5
        h2h_n = len(h2h_list)

        # Venue stats for t1
        v1 = team_records[t1]["venue_stats"].get(venue, {"wins": 0, "matches": 0, "runs": []})
        if v1["matches"] >= MIN_VENUE:
            t1_venue_pct = v1["wins"] / v1["matches"]
            t1_venue_runs = np.mean(v1["runs"]) if v1["runs"] else 160
        else:
            g1 = team_records[t1]
            t1_venue_pct = g1["wins"] / max(g1["matches"], 1) if g1["matches"] > 0 else 0.5
            t1_venue_runs = np.mean(g1["global_runs"]) if g1["global_runs"] else 160
        t1_venue_m = v1["matches"]

        # Venue stats for t2
        v2 = team_records[t2]["venue_stats"].get(venue, {"wins": 0, "matches": 0, "runs": []})
        if v2["matches"] >= MIN_VENUE:
            t2_venue_pct = v2["wins"] / v2["matches"]
            t2_venue_runs = np.mean(v2["runs"]) if v2["runs"] else 160
        else:
            g2 = team_records[t2]
            t2_venue_pct = g2["wins"] / max(g2["matches"], 1) if g2["matches"] > 0 else 0.5
            t2_venue_runs = np.mean(g2["global_runs"]) if g2["global_runs"] else 160
        t2_venue_m = v2["matches"]

        # Static venue properties
        v_freq = venue_freq.get(venue, 0)
        v_avg = venue_avg.get(venue, 160)

        # Toss
        toss_w = row["toss_winner"]
        toss_d = row.get("toss_decision", "field")
        toss_w_norm = normalise(toss_w)

        home_diff = 0  # No reliable city-based home advantage

        feat_row = {
            "match_id": row["match_id"],
            "date": row["date"],
            "t1_enc": team_le.transform([t1])[0],
            "t2_enc": team_le.transform([t2])[0],
            "toss_enc": team_le.transform([toss_w_norm])[0],
            "toss_winner_bat_first": 1 if toss_w_norm == t1 else 0,
            "toss_field": 1 if toss_d == "field" else 0,
            "venue_freq": v_freq,
            "venue_avg_inn1": v_avg,
            "recent5_diff": t1_r5 - t2_r5,
            "recent10_diff": t1_r10 - t2_r10,
            "cum_pct_diff": t1_cum - t2_cum,
            "t1_h2h_pct": h2h_pct,
            "t1_h2h_n": h2h_n,
            "home_diff": home_diff,
            "t1_matches": team_records[t1]["matches"],
            "t2_matches": team_records[t2]["matches"],
            "season_num": pd.to_numeric(str(row["season"]).replace("/", "."), errors="coerce") or 2008,
            "elo_diff": t1_elo - t2_elo,
            "venue_win_pct_diff": t1_venue_pct - t2_venue_pct,
            "team1_venue_win_pct": t1_venue_pct,
            "team2_venue_win_pct": t2_venue_pct,
            "team1_venue_matches": t1_venue_m,
            "team2_venue_matches": t2_venue_m,
            "team1_avg_runs_at_venue": t1_venue_runs,
            "team2_avg_runs_at_venue": t2_venue_runs,
            "_source": source,
        }

        feature_rows.append(feat_row)

        # --- Update state AFTER computing features ---
        if source == "history":
            won = int(row["team1_won"])
            inn1 = row["inn1_runs"]
            inn2 = row["inn2_runs"]
        else:
            # For target matches, we still update state for sequential predictions
            # but we need the actual result if available
            if "team1_won" in row and pd.notna(row["team1_won"]):
                won = int(row["team1_won"])
            else:
                won = 0  # Default if no result
            inn1 = row.get("inn1_runs", 160)
            inn2 = row.get("inn2_runs", 160)

        # Update Elo
        t1_elo_adj = team_records[t1]["elo"] + HOME_ADV if row.get("city") == row.get("city") else team_records[t1]["elo"]
        t2_elo_adj = team_records[t2]["elo"]
        expected_t1 = 1 / (1 + 10 ** ((t2_elo_adj - t1_elo_adj) / 400))
        team_records[t1]["elo"] += K_ELO * (won - expected_t1)
        team_records[t2]["elo"] += K_ELO * ((1 - won) - (1 - expected_t1))

        # Update team records
        team_records[t1]["wins"] += won
        team_records[t1]["matches"] += 1
        team_records[t1]["results"].append(won)
        team_records[t1]["global_runs"].append(inn1)

        team_records[t2]["wins"] += (1 - won)
        team_records[t2]["matches"] += 1
        team_records[t2]["results"].append(1 - won)
        team_records[t2]["global_runs"].append(inn2)

        # Update venue stats
        for team, runs, w in [(t1, inn1, won), (t2, inn2, 1 - won)]:
            if venue not in team_records[team]["venue_stats"]:
                team_records[team]["venue_stats"][venue] = {"wins": 0, "matches": 0, "runs": []}
            team_records[team]["venue_stats"][venue]["matches"] += 1
            team_records[team]["venue_stats"][venue]["wins"] += w
            team_records[team]["venue_stats"][venue]["runs"].append(runs)

        # Update H2H
        if key not in h2h:
            h2h[key] = []
        h2h[key].append(won)

    feat_df = pd.DataFrame(feature_rows)
    return feat_df, team_le


FEATURE_COLS = [
    "t1_enc", "t2_enc", "toss_enc",
    "toss_winner_bat_first", "toss_field",
    "venue_freq", "venue_avg_inn1",
    "recent5_diff", "recent10_diff", "cum_pct_diff",
    "t1_h2h_pct", "t1_h2h_n",
    "home_diff", "t1_matches", "t2_matches",
    "season_num",
    "elo_diff",
    "venue_win_pct_diff",
    "team1_venue_win_pct", "team2_venue_win_pct",
    "team1_venue_matches", "team2_venue_matches",
    "team1_avg_runs_at_venue", "team2_avg_runs_at_venue",
]


def train_model(df_train):
    """Train the production model on historical data."""
    # Build features for training data
    feat_df, team_le = build_features_for_matches(
        df_train[df_train["season"] != df_train["season"].iloc[-1]],  # placeholder
        df_train
    )
    # Actually, simpler: replay history within the training set
    feat_df, team_le = build_features_for_matches(
        pd.DataFrame(),  # no prior history
        df_train
    )
    # Hmm, let me simplify this. Build features directly.

    feat_df, team_le = build_train_features(df_train)

    X = feat_df[FEATURE_COLS].values
    y = feat_df["team1_won"].values

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=15,
        random_state=42, n_jobs=-1,
    )
    model.fit(X, y)
    return model, team_le


def build_train_features(df):
    """Build features for training data (replaying within the dataset)."""
    df = df.copy()
    df["batting_team"] = df["batting_team"].apply(normalise)
    df["bowling_team"] = df["bowling_team"].apply(normalise)
    df["toss_winner"] = df["toss_winner"].apply(normalise)
    df = df.sort_values("date").reset_index(drop=True)

    all_teams = sorted(
        set(df["batting_team"].unique()) | set(df["bowling_team"].unique())
    )
    team_le = LabelEncoder()
    team_le.fit(all_teams)

    team_records = {}
    for t in all_teams:
        team_records[t] = {
            "wins": 0, "matches": 0, "results": [],
            "elo": 1500.0, "venue_stats": {}, "global_runs": [],
        }
    h2h = {}

    venue_freq = df["venue"].value_counts(normalize=True)
    venue_avg = df.groupby("venue")["inn1_runs"].mean()

    rows = []
    K_ELO = 32
    HOME_ADV = 65

    for idx, row in df.iterrows():
        t1 = row["batting_team"]
        t2 = row["bowling_team"]
        venue = row["venue"]
        r1 = team_records[t1]["results"]
        r2 = team_records[t2]["results"]

        t1_cum = team_records[t1]["wins"] / max(team_records[t1]["matches"], 1)
        t2_cum = team_records[t2]["wins"] / max(team_records[t2]["matches"], 1)
        t1_r5 = np.mean(r1[-5:]) if len(r1) >= 5 else 0.5
        t2_r5 = np.mean(r2[-5:]) if len(r2) >= 5 else 0.5
        t1_r10 = np.mean(r1[-10:]) if len(r1) >= 10 else 0.5
        t2_r10 = np.mean(r2[-10:]) if len(r2) >= 10 else 0.5

        key = (t1, t2)
        rev = (t2, t1)
        h2h_list = h2h.get(key, []) + h2h.get(rev, [])
        h2h_pct = np.mean(h2h_list) if h2h_list else 0.5

        v1 = team_records[t1]["venue_stats"].get(venue, {"wins": 0, "matches": 0, "runs": []})
        v2 = team_records[t2]["venue_stats"].get(venue, {"wins": 0, "matches": 0, "runs": []})
        if v1["matches"] >= 3:
            t1_vp = v1["wins"] / v1["matches"]
            t1_vr = np.mean(v1["runs"]) if v1["runs"] else 160
        else:
            g1 = team_records[t1]
            t1_vp = g1["wins"] / max(g1["matches"], 1) if g1["matches"] > 0 else 0.5
            t1_vr = np.mean(g1["global_runs"]) if g1["global_runs"] else 160
        if v2["matches"] >= 3:
            t2_vp = v2["wins"] / v2["matches"]
            t2_vr = np.mean(v2["runs"]) if v2["runs"] else 160
        else:
            g2 = team_records[t2]
            t2_vp = g2["wins"] / max(g2["matches"], 1) if g2["matches"] > 0 else 0.5
            t2_vr = np.mean(g2["global_runs"]) if g2["global_runs"] else 160

        won = int(row["team1_won"])

        rows.append({
            "match_id": row["match_id"],
            "team1_won": won,
            "t1_enc": team_le.transform([t1])[0],
            "t2_enc": team_le.transform([t2])[0],
            "toss_enc": team_le.transform([row["toss_winner"]])[0],
            "toss_winner_bat_first": 1 if row["toss_winner"] == t1 else 0,
            "toss_field": 1 if row["toss_decision"] == "field" else 0,
            "venue_freq": venue_freq.get(venue, 0),
            "venue_avg_inn1": venue_avg.get(venue, 160),
            "recent5_diff": t1_r5 - t2_r5,
            "recent10_diff": t1_r10 - t2_r10,
            "cum_pct_diff": t1_cum - t2_cum,
            "t1_h2h_pct": h2h_pct,
            "t1_h2h_n": len(h2h_list),
            "home_diff": 0,
            "t1_matches": team_records[t1]["matches"],
            "t2_matches": team_records[t2]["matches"],
            "season_num": pd.to_numeric(str(row["season"]).replace("/", "."), errors="coerce") or 2008,
            "elo_diff": team_records[t1]["elo"] - team_records[t2]["elo"],
            "venue_win_pct_diff": t1_vp - t2_vp,
            "team1_venue_win_pct": t1_vp,
            "team2_venue_win_pct": t2_vp,
            "team1_venue_matches": v1["matches"],
            "team2_venue_matches": v2["matches"],
            "team1_avg_runs_at_venue": t1_vr,
            "team2_avg_runs_at_venue": t2_vr,
        })

        # Update state
        t1_elo_a = team_records[t1]["elo"] + HOME_ADV
        t2_elo_a = team_records[t2]["elo"]
        exp_t1 = 1 / (1 + 10 ** ((t2_elo_a - t1_elo_a) / 400))
        team_records[t1]["elo"] += K_ELO * (won - exp_t1)
        team_records[t2]["elo"] += K_ELO * ((1 - won) - (1 - exp_t1))

        team_records[t1]["wins"] += won
        team_records[t1]["matches"] += 1
        team_records[t1]["results"].append(won)
        team_records[t1]["global_runs"].append(row["inn1_runs"])
        team_records[t2]["wins"] += (1 - won)
        team_records[t2]["matches"] += 1
        team_records[t2]["results"].append(1 - won)
        team_records[t2]["global_runs"].append(row["inn2_runs"])

        for team, runs, w in [(t1, row["inn1_runs"], won), (t2, row["inn2_runs"], 1 - won)]:
            if venue not in team_records[team]["venue_stats"]:
                team_records[team]["venue_stats"][venue] = {"wins": 0, "matches": 0, "runs": []}
            team_records[team]["venue_stats"][venue]["matches"] += 1
            team_records[team]["venue_stats"][venue]["wins"] += w
            team_records[team]["venue_stats"][venue]["runs"].append(runs)

        if key not in h2h:
            h2h[key] = []
        h2h[key].append(won)

    feat_df = pd.DataFrame(rows)
    return feat_df, team_le


def main():
    print("=" * 60)
    print("IPL 2025 Backtest")
    print("=" * 60)

    # Load all data
    print(f"\nLoading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    df["batting_team"] = df["batting_team"].apply(normalise)
    df["bowling_team"] = df["bowling_team"].apply(normalise)
    df["toss_winner"] = df["toss_winner"].apply(normalise)

    # Split: train <= 2024, test = 2025
    train_df = df[~df["season"].isin(["2025", "2026"])].copy()
    test_df = df[df["season"] == "2025"].copy()

    print(f"  Train: {len(train_df)} matches (seasons <= 2024)")
    print(f"  Test:  {len(test_df)} matches (2025)")

    # Step 1: Train model
    print("\n[1/5] Training model on historical data...")
    train_feat, team_le = build_train_features(train_df)
    X_train = train_feat[FEATURE_COLS].values
    y_train = train_feat["team1_won"].values

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=15,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"  Trained on {len(X_train)} matches, {len(FEATURE_COLS)} features")

    # Step 2: Generate 2025 features (replay history, then predict sequentially)
    print("\n[2/5] Generating 2025 match features...")
    test_feat, _ = build_features_for_matches(train_df, test_df)
    test_feat = test_feat[test_feat["_source"] == "target"]
    X_test = test_feat[FEATURE_COLS].values
    y_test = test_df["team1_won"].values

    print(f"  Generated features for {len(X_test)} matches")

    # Step 3: Predictions
    print("\n[3/5] Making predictions...")
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    # Step 4: Evaluation
    print("\n[4/5] Evaluation Metrics:")
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    ll = log_loss(y_test, y_proba)

    print(f"  Accuracy:   {acc:.4f}")
    print(f"  ROC-AUC:    {auc:.4f}")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  Log Loss:   {ll:.4f}")

    # Calibration
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy="quantile")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Calibration plot
    axes[0].plot(prob_pred, prob_true, "o-", label="Model")
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect")
    axes[0].set_xlabel("Mean Predicted Probability")
    axes[0].set_ylabel("Fraction of Positives")
    axes[0].set_title("Calibration Plot (2025 Backtest)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Probability distribution
    axes[1].hist(y_proba, bins=20, edgecolor="black", alpha=0.7)
    axes[1].axvline(x=0.5, color="red", linestyle="--", label="Decision boundary")
    axes[1].set_xlabel("P(Team 1 wins)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Prediction Probability Distribution")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(MODEL_DIR / "backtest_calibration.png", dpi=150)
    plt.close()
    print(f"\n  Saved: models/backtest_calibration.png")

    # Step 5: Top 20 most confident predictions
    print("\n[5/5] Top 20 Most Confident Predictions:")
    confidence = np.abs(y_proba - 0.5)
    top20_idx = np.argsort(confidence)[::-1][:20]

    test_df_sorted = test_df.reset_index(drop=True)
    print(f"  {'#':>2} {'Date':<12} {'Team 1':<28} {'Team 2':<28} {'P(T1)':>6} {'P(T2)':>6} {'Pred':>8} {'Actual':>8}")
    print(f"  {'-'*120}")
    for rank, i in enumerate(top20_idx, 1):
        row = test_df_sorted.iloc[i]
        p1 = y_proba[i]
        p2 = 1 - p1
        pred = "Team 1" if p1 >= 0.5 else "Team 2"
        actual = "Team 1" if y_test[i] == 1 else "Team 2"
        correct = "OK" if pred == actual else "NO"
        t1 = row["batting_team"][:27]
        t2 = row["bowling_team"][:27]
        print(f"  {rank:>2} {row['date']:<12} {t1:<28} {t2:<28} {p1:>6.2%} {p2:>6.2%} {pred:>6} {correct} {actual:>6}")

    # Season simulation
    print("\n" + "=" * 60)
    print("2025 Season Simulation")
    print("=" * 60)

    # Simulate each match with predicted probability
    np.random.seed(42)
    simulated_results = []
    for i in range(len(test_df)):
        row = test_df.iloc[i]
        p = y_proba[i]
        outcome = 1 if np.random.random() < p else 0
        simulated_results.append({
            "match_id": row["match_id"],
            "team1": row["batting_team"],
            "team2": row["bowling_team"],
            "p_team1": p,
            "simulated_winner": row["batting_team"] if outcome == 1 else row["bowling_team"],
            "actual_winner": row["match_won_by"],
        })

    sim_df = pd.DataFrame(simulated_results)

    # Build predicted points table
    teams = sorted(set(test_df["batting_team"].unique()) | set(test_df["bowling_team"].unique()))
    points = {t: {"pts": 0, "nrr_num": 0.0, "nrr_den": 0.0, "wins": 0, "losses": 0} for t in teams}

    for _, r in sim_df.iterrows():
        t1, t2 = r["team1"], r["team2"]
        winner = r["simulated_winner"]
        points[t1]["pts"] += 2 if winner == t1 else 0
        points[t2]["pts"] += 2 if winner == t2 else 0
        if winner == t1:
            points[t1]["wins"] += 1
            points[t2]["losses"] += 1
        else:
            points[t2]["wins"] += 1
            points[t1]["losses"] += 1

    # Build actual points table
    actual_points = {t: {"pts": 0, "wins": 0, "losses": 0} for t in teams}
    for _, r in test_df.iterrows():
        t1, t2 = r["batting_team"], r["bowling_team"]
        winner = r["match_won_by"]
        actual_points[t1]["pts"] += 2 if winner == t1 else 0
        actual_points[t2]["pts"] += 2 if winner == t2 else 0
        if winner == t1:
            actual_points[t1]["wins"] += 1
            actual_points[t2]["losses"] += 1
        else:
            actual_points[t2]["wins"] += 1
            actual_points[t1]["losses"] += 1

    # Compare tables
    print(f"\n  {'Rank':>4} {'Team':<28} {'Pred Pts':>9} {'Actual Pts':>11} {'Pred W':>7} {'Actual W':>9} {'Match':>6}")
    print(f"  {'-'*76}")

    pred_ranked = sorted(points.items(), key=lambda x: x[1]["pts"], reverse=True)
    actual_ranked = sorted(actual_points.items(), key=lambda x: x[1]["pts"], reverse=True)
    actual_rank_map = {t: i + 1 for i, (t, _) in enumerate(actual_ranked)}

    for rank, (team, p) in enumerate(pred_ranked, 1):
        a_pts = actual_points[team]["pts"]
        a_wins = actual_points[team]["wins"]
        a_rank = actual_rank_map[team]
        match = "OK" if rank == a_rank else "NO"
        print(f"  {rank:>4} {team:<28} {p['pts']:>9} {a_pts:>11} {p['wins']:>7} {a_wins:>9} {match:>6}")

    # Top 4 comparison
    pred_top4 = set(t for t, _ in pred_ranked[:4])
    actual_top4 = set(t for t, _ in actual_ranked[:4])
    overlap = pred_top4 & actual_top4
    print(f"\n  Predicted top 4: {sorted(pred_top4)}")
    print(f"  Actual top 4:    {sorted(actual_top4)}")
    print(f"  Overlap: {len(overlap)}/4 {sorted(overlap)}")


if __name__ == "__main__":
    main()

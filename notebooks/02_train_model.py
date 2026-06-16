"""
IPL Tournament Winner Prediction - Phase 2: Model Training
============================================================
Loads processed match data, trains classifiers to predict match winners
using pre-match features, evaluates performance, and saves the model.

Chronological train/test split: trains on earlier seasons, tests on later
seasons to avoid temporal data leakage.

Uses scikit-learn RandomForest.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, precision_score, recall_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "matches.csv"
ELO_PATH = BASE_DIR / "data" / "processed" / "matches_with_elo.csv"
VENUE_PATH = BASE_DIR / "data" / "processed" / "matches_with_venue.csv"
MOMENTUM_PATH = BASE_DIR / "data" / "processed" / "matches_with_momentum.csv"
SCORING_PATH = BASE_DIR / "data" / "processed" / "matches_with_scoring.csv"
MODEL_DIR = BASE_DIR / "models"
FEATURES_PATH = MODEL_DIR / "feature_config.json"
EVAL_PATH = MODEL_DIR / "evaluation.json"

TEAM_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Deccan Chargers": "Sunrisers Hyderabad",
}


def normalise(name):
    return TEAM_MAP.get(name, name)


def prepare_features(df, has_elo=False, has_venue=False, has_momentum=False, has_scoring=False):
    """Build pre-match features."""
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
    df["t1_enc"] = team_le.transform(df["batting_team"])
    df["t2_enc"] = team_le.transform(df["bowling_team"])
    df["toss_enc"] = team_le.transform(df["toss_winner"])

    venue_freq = df["venue"].value_counts(normalize=True)
    df["venue_freq"] = df["venue"].map(venue_freq)

    df["season_num"] = pd.to_numeric(
        df["season"].str.replace("/", "."), errors="coerce"
    ).fillna(2008)

    df["t1_home"] = (df["batting_team"] == df["city"]).astype(int)
    df["t2_home"] = (df["bowling_team"] == df["city"]).astype(int)

    df["toss_winner_bat_first"] = (df["toss_winner"] == df["batting_team"]).astype(int)
    df["toss_field"] = (df["toss_decision"] == "field").astype(int)

    venue_avg = df.groupby("venue")["inn1_runs"].mean()
    df["venue_avg_inn1"] = df["venue"].map(venue_avg)

    # Rolling features (chronological — only prior matches)
    team_records = {}
    for t in all_teams:
        team_records[t] = {"wins": 0, "matches": 0, "results": []}

    h2h = {}
    t1_recent5, t2_recent5 = [], []
    t1_recent10, t2_recent10 = [], []
    t1_cum_pct, t2_cum_pct = [], []
    t1_matches, t2_matches = [], []
    t1_h2h_pct, t1_h2h_n = [], []

    for idx, row in df.iterrows():
        t1, t2 = row["batting_team"], row["bowling_team"]

        r1 = team_records[t1]["results"]
        t1_recent5.append(np.mean(r1[-5:]) if r1 else 0.5)
        t1_recent10.append(np.mean(r1[-10:]) if r1 else 0.5)
        t1_cum_pct.append(team_records[t1]["wins"] / max(team_records[t1]["matches"], 1))
        t1_matches.append(team_records[t1]["matches"])

        r2 = team_records[t2]["results"]
        t2_recent5.append(np.mean(r2[-5:]) if r2 else 0.5)
        t2_recent10.append(np.mean(r2[-10:]) if r2 else 0.5)
        t2_cum_pct.append(team_records[t2]["wins"] / max(team_records[t2]["matches"], 1))
        t2_matches.append(team_records[t2]["matches"])

        key = (t1, t2)
        rev_key = (t2, t1)
        h2h_list = h2h.get(key, []) + h2h.get(rev_key, [])
        t1_h2h_pct.append(np.mean(h2h_list) if h2h_list else 0.5)
        t1_h2h_n.append(len(h2h_list))

        won = int(row["team1_won"])
        team_records[t1]["wins"] += won
        team_records[t1]["matches"] += 1
        team_records[t1]["results"].append(won)
        team_records[t2]["wins"] += (1 - won)
        team_records[t2]["matches"] += 1
        team_records[t2]["results"].append(1 - won)
        if key not in h2h:
            h2h[key] = []
        h2h[key].append(won)

    df["t1_recent5"] = t1_recent5
    df["t2_recent5"] = t2_recent5
    df["t1_recent10"] = t1_recent10
    df["t2_recent10"] = t2_recent10
    df["t1_cum_pct"] = t1_cum_pct
    df["t2_cum_pct"] = t2_cum_pct
    df["t1_matches"] = t1_matches
    df["t2_matches"] = t2_matches
    df["t1_h2h_pct"] = t1_h2h_pct
    df["t1_h2h_n"] = t1_h2h_n

    df["recent5_diff"] = df["t1_recent5"] - df["t2_recent5"]
    df["recent10_diff"] = df["t1_recent10"] - df["t2_recent10"]
    df["cum_pct_diff"] = df["t1_cum_pct"] - df["t2_cum_pct"]
    df["home_diff"] = df["t1_home"] - df["t2_home"]
    df["matches_diff"] = df["t1_matches"].astype(float) - df["t2_matches"].astype(float)

    feature_cols = [
        "t1_enc", "t2_enc", "toss_enc",
        "toss_winner_bat_first", "toss_field",
        "venue_freq", "venue_avg_inn1",
        "recent5_diff", "recent10_diff", "cum_pct_diff",
        "t1_h2h_pct", "t1_h2h_n",
        "home_diff", "t1_matches", "t2_matches",
        "season_num",
    ]
    if has_elo and "elo_diff" in df.columns:
        feature_cols.append("elo_diff")
        print(f"  + Elo features enabled (elo_diff)")
    if has_venue and "venue_win_pct_diff" in df.columns:
        feature_cols.extend([
            "venue_win_pct_diff",
            "team1_venue_win_pct", "team2_venue_win_pct",
            "team1_venue_matches", "team2_venue_matches",
            "team1_avg_runs_at_venue", "team2_avg_runs_at_venue",
        ])
        print(f"  + Venue features enabled (7 features)")
    if has_momentum and "last3_diff" in df.columns:
        feature_cols.extend([
            "last3_diff", "last5_diff", "streak_diff",
            "team1_last3_win_pct", "team2_last3_win_pct",
            "team1_last5_win_pct", "team2_last5_win_pct",
            "team1_current_win_streak", "team2_current_win_streak",
            "team1_matches_since_loss", "team2_matches_since_loss",
        ])
        print(f"  + Momentum features enabled (11 features)")
    if has_scoring and "nrr_diff" in df.columns:
        feature_cols.extend([
            "runs_scored_diff", "runs_conceded_diff",
            "nrr_diff",
            "team1_nrr", "team2_nrr",
            "team1_avg_runs_scored_last10", "team2_avg_runs_scored_last10",
            "team1_avg_runs_conceded_last10", "team2_avg_runs_conceded_last10",
            "team1_powerplay_run_rate", "team2_powerplay_run_rate",
            "team1_death_over_run_rate", "team2_death_over_run_rate",
        ])
        print(f"  + Scoring/NRR features enabled (13 features)")

    df = df.dropna(subset=feature_cols + ["team1_won"])
    X = df[feature_cols].values
    y = df["team1_won"].values

    return X, y, feature_cols, team_le


def train_and_evaluate(X, y, feature_cols, team_le, df):
    """Train with CHRONOLOGICAL split: earlier seasons train, later seasons test."""
    seasons = pd.to_numeric(
        df["season"].str.replace("/", "."), errors="coerce"
    ).fillna(2008)

    # Last 3 seasons as test
    unique_seasons = sorted(seasons.unique())
    threshold = unique_seasons[-3]

    train_mask = seasons < threshold
    test_mask = seasons >= threshold

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"\nChronological split (threshold: season {threshold})")
    print(f"  Train: {len(X_train)} (seasons < {threshold})")
    print(f"  Test:  {len(X_test)}  (seasons >= {threshold})")
    print(f"  Train win rate: {y_train.mean():.1%} | Test win rate: {y_test.mean():.1%}")

    model = RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=15,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)

    print(f"\nRandomForest — Acc: {acc:.4f} | AUC: {auc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, pred, target_names=["Team 2 won", "Team 1 won"]))

    cm = confusion_matrix(y_test, pred)
    print("Confusion Matrix:")
    print(f"  TN={cm[0][0]}  FN={cm[0][1]}")
    print(f"  FP={cm[1][0]}  TP={cm[1][1]}")

    # Time-series cross-validation (expanding window)
    print("\nTime-Series Cross-Validation (5-fold expanding window):")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_aucs = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]
        m = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=15,
            random_state=42, n_jobs=-1,
        )
        m.fit(X_tr, y_tr)
        val_proba = m.predict_proba(X_val)[:, 1]
        if len(np.unique(y_val)) > 1:
            fold_auc = roc_auc_score(y_val, val_proba)
            cv_aucs.append(fold_auc)
            print(f"  Fold {fold+1}: AUC = {fold_auc:.4f}")
    cv_mean = np.mean(cv_aucs) if cv_aucs else 0.5
    cv_std = np.std(cv_aucs) if cv_aucs else 0.0
    print(f"  Mean AUC: {cv_mean:.4f} +/- {cv_std:.4f}")

    importances = model.feature_importances_
    feat_imp = pd.DataFrame(
        {"feature": feature_cols, "importance": importances}
    ).sort_values("importance", ascending=False)

    print("\nFeature Importance:")
    for _, r in feat_imp.iterrows():
        bar = "#" * int(r["importance"] * 50)
        print(f"  {r['feature']:<25} {r['importance']:.4f} {bar}")

    return {
        "model": model, "name": "RandomForest",
        "accuracy": acc, "auc": auc,
        "precision": prec, "recall": rec,
        "cv_auc_mean": cv_mean, "cv_auc_std": cv_std,
        "feature_importance": feat_imp, "y_test": y_test,
        "y_pred": pred, "y_proba": proba,
        "team_le": team_le, "feature_cols": feature_cols,
        "confusion_matrix": cm,
        "test_season_threshold": threshold,
    }


def save_artifacts(result):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "random_forest_model.joblib"
    encoder_path = MODEL_DIR / "team_encoder.joblib"

    joblib.dump(result["model"], model_path)
    joblib.dump(result["team_le"], encoder_path)

    with open(FEATURES_PATH, "w") as f:
        json.dump(
            {"feature_cols": result["feature_cols"], "model_name": result["name"]},
            f, indent=2,
        )

    cm = result["confusion_matrix"]
    with open(EVAL_PATH, "w") as f:
        json.dump({
            "model": result["name"],
            "accuracy": round(result["accuracy"], 4),
            "roc_auc": round(result["auc"], 4),
            "precision": round(result["precision"], 4),
            "recall": round(result["recall"], 4),
            "cv_auc_mean": round(result.get("cv_auc_mean", 0), 4),
            "cv_auc_std": round(result.get("cv_auc_std", 0), 4),
            "confusion_matrix": {
                "TN": int(cm[0][0]), "FN": int(cm[0][1]),
                "FP": int(cm[1][0]), "TP": int(cm[1][1]),
            },
        }, f, indent=2)

    print(f"\nSaved model: {model_path}")


def plot_results(result):
    # Feature importance
    fig, ax = plt.subplots(figsize=(10, 7))
    top = result["feature_importance"].head(12).iloc[::-1]
    ax.barh(top["feature"], top["importance"], color="#4472C4")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance - {result['name']}")
    plt.tight_layout()
    fig.savefig(MODEL_DIR / "feature_importance.png", dpi=150)
    plt.close()

    # ROC curve
    fig, ax = plt.subplots(figsize=(7, 5))
    if len(np.unique(result["y_test"])) > 1:
        from sklearn.metrics import RocCurveDisplay
        RocCurveDisplay.from_predictions(
            result["y_test"], result["y_proba"], ax=ax,
            name=f"AUC={result['auc']:.4f}"
        )
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(MODEL_DIR / "roc_curve.png", dpi=150)
    plt.close()

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = result["confusion_matrix"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Team 2", "Team 1"],
                yticklabels=["Team 2", "Team 1"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    fig.savefig(MODEL_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def main():
    print("=" * 60)
    print("IPL Model Training - Phase 2")
    print("=" * 60)

    print(f"\nLoading data...")
    if SCORING_PATH.exists():
        df = pd.read_csv(SCORING_PATH)
        print(f"  Loaded scoring features ({len(df)} matches)")
        has_scoring = True
        has_momentum = "last3_diff" in df.columns
        has_venue = "venue_win_pct_diff" in df.columns
        has_elo = "elo_diff" in df.columns
    elif MOMENTUM_PATH.exists():
        df = pd.read_csv(MOMENTUM_PATH)
        print(f"  Loaded momentum features ({len(df)} matches)")
        has_scoring = False
        has_momentum = True
        has_venue = "venue_win_pct_diff" in df.columns
        has_elo = "elo_diff" in df.columns
    elif VENUE_PATH.exists():
        df = pd.read_csv(VENUE_PATH)
        print(f"  Loaded venue features ({len(df)} matches)")
        has_scoring = False
        has_momentum = False
        has_venue = True
        has_elo = "elo_diff" in df.columns
    elif ELO_PATH.exists():
        df = pd.read_csv(ELO_PATH)
        print(f"  Loaded Elo features ({len(df)} matches)")
        has_scoring = False
        has_momentum = False
        has_venue = False
        has_elo = True
    else:
        df = pd.read_csv(DATA_PATH)
        print(f"  Loaded base data ({len(df)} matches)")
        has_scoring = False
        has_momentum = False
        has_venue = False
        has_elo = False

    print("\nPreparing features...")
    X, y, feature_cols, team_le = prepare_features(
        df, has_elo=has_elo, has_venue=has_venue,
        has_momentum=has_momentum, has_scoring=has_scoring
    )
    print(f"  Features: {len(feature_cols)} | Samples: {len(X)}")

    print("\nTraining...")
    result = train_and_evaluate(X, y, feature_cols, team_le, df)

    print("\nSaving...")
    save_artifacts(result)
    plot_results(result)

    print(f"\nDone — {result['name']}")
    print(f"  Accuracy:     {result['accuracy']:.4f}")
    print(f"  ROC-AUC:      {result['auc']:.4f}")
    print(f"  Precision:    {result['precision']:.4f}")
    print(f"  Recall:       {result['recall']:.4f}")
    print(f"  TS-CV AUC:    {result.get('cv_auc_mean', 0):.4f} +/- {result.get('cv_auc_std', 0):.4f}")


if __name__ == "__main__":
    main()

"""
IPL Model Diagnostics - Phase 2.5
===================================
Audits the training pipeline for correctness:
target encoding, class balance, feature leakage, and model comparison.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import TimeSeriesSplit

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "matches.csv"

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


# ============================================================
# SECTION 1: Target Validation
# ============================================================

def verify_target_encoding(df):
    """Verify target variable encoding and positive class assignment."""
    print("=" * 60)
    print("1. TARGET ENCODING VERIFICATION")
    print("=" * 60)

    # Check the target column
    print(f"\nTarget column: '{TARGET}'")
    print(f"  Unique values: {sorted(df[TARGET].unique())}")
    print(f"  Dtype: {df[TARGET].dtype}")
    print(f"  Value counts:\n{df[TARGET].value_counts().sort_index()}")

    # Verify what '1' means
    won_when_1 = df[df[TARGET] == 1][["batting_team", "match_won_by"]].head(10)
    print(f"\n  Sample: When {TARGET}=1, did batting_team win?")
    for _, row in won_when_1.iterrows():
        match = "YES" if row["batting_team"] == row["match_won_by"] else "NO"
        print(f"    {row['batting_team']} vs ? -> {row['match_won_by']} | {match}")

    correct = (df[df[TARGET] == 1]["batting_team"] == df[df[TARGET] == 1]["match_won_by"]).all()
    print(f"\n  Verification: {TARGET}=1 always means batting_team won: {correct}")

    # Check class balance overall
    pos_rate = df[TARGET].mean()
    print(f"\n  Overall positive rate ({TARGET}=1): {pos_rate:.1%}")
    print(f"  Baseline: always predict 0 -> accuracy = {1 - pos_rate:.1%}")
    print(f"  Baseline: always predict 1 -> accuracy = {pos_rate:.1%}")

    # Class balance by season
    print("\n  Class balance by season:")
    season_balance = df.groupby("season").agg(
        total=(TARGET, "count"),
        positives=(TARGET, "sum"),
        rate=(TARGET, "mean"),
    )
    print(f"  {'Season':<15} {'Total':>6} {'Pos(1)':>7} {'Rate':>7}")
    print(f"  {'-'*37}")
    for season, row in season_balance.iterrows():
        print(f"  {season:<15} {int(row['total']):>6} {int(row['positives']):>7} {row['rate']:>7.1%}")

    # Check if recent seasons have shifted
    recent = season_balance.tail(5)
    print(f"\n  Last 5 seasons avg rate: {recent['rate'].mean():.1%}")
    old = season_balance.head(10)
    print(f"  First 10 seasons avg rate: {old['rate'].mean():.1%}")
    shift = recent["rate"].mean() - old["rate"].mean()
    print(f"  Shift: {shift:+.1%} (positive = more bat-first wins recently)")

    return correct, pos_rate


def check_probability_inversion(X, y, df):
    """Check whether model probabilities are inverted."""
    print("\n" + "=" * 60)
    print("2. PROBABILITY INVERSION CHECK")
    print("=" * 60)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Quick model on random split for diagnostic only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    # AUC should be > 0.5 if probabilities are correctly aligned
    auc = roc_auc_score(y_test, proba)
    print(f"\n  Quick RF model AUC (positive={TARGET}): {auc:.4f}")

    if auc < 0.5:
        print(f"  WARNING: AUC < 0.5 — probabilities may be inverted!")
        print(f"  The model predicts OPPOSITE of {TARGET} correctly.")
        print(f"  Suggested fix: use 1 - proba or flip the target.")
    else:
        print(f"  OK: AUC > 0.5 — probabilities correctly aligned with {TARGET}.")

    # Calibration check
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    print(f"\n  Calibration (expected prob vs actual prob):")
    for pt, pp in zip(prob_pred, prob_true):
        bar_actual = "#" * int(pp * 30)
        print(f"    Predicted {pp:.1%} -> Actual {pt:.1%} {bar_actual}")

    # Brier score
    brier = brier_score_loss(y_test, proba)
    base_rate = y_test.mean()
    baseline_brier = base_rate * (1 - base_rate)
    print(f"\n  Brier score: {brier:.4f} (lower is better, perfect=0, baseline={baseline_brier:.4f})")

    return auc


# ============================================================
# Run diagnostics
# ============================================================

def main():
    print("IPL Model Diagnostics - Phase 2.5")
    print("=" * 60)

    print(f"\nLoading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} matches loaded")

    # Step 1: Target verification
    correct, pos_rate = verify_target_encoding(df)

    # Build quick features for probability check
    print("\n\n[Building minimal features for probability check...]")
    df2 = df.copy()
    df2["batting_team"] = df2["batting_team"].apply(normalise)
    df2["bowling_team"] = df2["bowling_team"].apply(normalise)
    df2["toss_winner"] = df2["toss_winner"].apply(normalise)
    df2 = df2.sort_values("date").reset_index(drop=True)

    all_teams = sorted(set(df2["batting_team"].unique()) | set(df2["bowling_team"].unique()))
    team_le = LabelEncoder()
    team_le.fit(all_teams)
    df2["t1_enc"] = team_le.transform(df2["batting_team"])
    df2["t2_enc"] = team_le.transform(df2["bowling_team"])
    df2["toss_enc"] = team_le.transform(df2["toss_winner"])

    # Quick rolling features
    team_records = {t: {"wins": 0, "matches": 0, "results": []} for t in all_teams}
    h2h = {}
    t1_cum, t2_cum, t1_m, t2_m = [], [], [], []
    h2h_pct, h2h_n = [], []

    for idx, row in df2.iterrows():
        t1, t2 = row["batting_team"], row["bowling_team"]
        r1 = team_records[t1]["results"]
        t1_cum.append(team_records[t1]["wins"] / max(team_records[t1]["matches"], 1) if r1 else 0.5)
        t2_cum.append(team_records[t2]["wins"] / max(team_records[t2]["matches"], 1) if team_records[t2]["results"] else 0.5)
        t1_m.append(team_records[t1]["matches"])
        t2_m.append(team_records[t2]["matches"])

        key = (t1, t2)
        rev = (t2, t1)
        hl = h2h.get(key, []) + h2h.get(rev, [])
        h2h_pct.append(np.mean(hl) if hl else 0.5)
        h2h_n.append(len(hl))

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

    df2["cum_diff"] = [a - b for a, b in zip(t1_cum, t2_cum)]
    df2["t1_matches"] = t1_m
    df2["t2_matches"] = t2_m
    df2["h2h_pct"] = h2h_pct
    df2["h2h_n"] = h2h_n

    feat_cols = ["t1_enc", "t2_enc", "toss_enc", "cum_diff", "t1_matches", "t2_matches", "h2h_pct", "h2h_n"]
    df2 = df2.dropna(subset=feat_cols + [TARGET])
    X = df2[feat_cols].values
    y = df2[TARGET].values

    # Step 2: Probability inversion check
    auc = check_probability_inversion(X, y, df2)

    print("\n" + "=" * 60)
    print("DIAGNOSTICS SUMMARY")
    print("=" * 60)
    print(f"  Target encoding correct: {correct}")
    print(f"  Overall positive rate: {pos_rate:.1%}")
    print(f"  Probability inversion: {'YES - INVERTED' if auc < 0.5 else 'NO - OK'}")
    print(f"  Quick model AUC: {auc:.4f}")


if __name__ == "__main__":
    main()

"""
IPL Tournament Simulation — Phase 4
=====================================
Monte Carlo simulation of IPL 2026 using the Elo+Venue production model.

DISCLAIMER: Research simulator. Model performed below random on IPL 2025 backtest.
Results are for educational purposes only.

Production model: Elo + Venue (24 features), RandomForest
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

RESEARCH_NOTICE = """Research Notice

The underlying match prediction model achieved:
- 40.5% accuracy on IPL 2025
- 0.425 ROC-AUC
- Worse-than-random performance on out-of-sample data

Simulation outputs are provided for educational and software engineering purposes and should not be interpreted as reliable forecasts."""

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "matches.csv"
SCHEDULE_PATH = BASE_DIR / "data" / "2026_schedule.csv"
OUTPUT_DIR = BASE_DIR / "output"

TEAM_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
    "Deccan Chargers": "Sunrisers Hyderabad",
}

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


def normalise(name):
    return TEAM_MAP.get(name, name)


def build_train_features(df):
    """Build features for training data with chronological replay."""
    df = df.copy()
    df["batting_team"] = df["batting_team"].apply(normalise)
    df["bowling_team"] = df["bowling_team"].apply(normalise)
    df["toss_winner"] = df["toss_winner"].apply(normalise)
    df = df.sort_values("date").reset_index(drop=True)

    all_teams = sorted(set(df["batting_team"].unique()) | set(df["bowling_team"].unique()))
    team_le = LabelEncoder()
    team_le.fit(all_teams)

    rec = {t: {"w":0,"m":0,"r":[],"elo":1500.0,"vs":{},"gr":[]} for t in all_teams}
    h2h = {}
    vfreq = df["venue"].value_counts(normalize=True)
    vavg = df.groupby("venue")["inn1_runs"].mean()

    rows = []
    for _, row in df.iterrows():
        t1, t2, ven = row["batting_team"], row["bowling_team"], row["venue"]
        r1, r2 = rec[t1]["r"], rec[t2]["r"]
        c1 = rec[t1]["w"]/max(rec[t1]["m"],1)
        c2 = rec[t2]["w"]/max(rec[t2]["m"],1)
        w5_1 = np.mean(r1[-5:]) if len(r1)>=5 else 0.5
        w5_2 = np.mean(r2[-5:]) if len(r2)>=5 else 0.5
        w10_1 = np.mean(r1[-10:]) if len(r1)>=10 else 0.5
        w10_2 = np.mean(r2[-10:]) if len(r2)>=10 else 0.5

        hl = h2h.get((t1,t2),[]) + h2h.get((t2,t1),[])
        hp = np.mean(hl) if hl else 0.5

        def gv(t):
            vs = rec[t]["vs"].get(ven, {"w":0,"m":0,"r":[]})
            if vs["m"]>=3: return vs["w"]/vs["m"], np.mean(vs["r"]) if vs["r"] else 160
            g = rec[t]
            return g["w"]/max(g["m"],1) if g["m"]>0 else 0.5, np.mean(g["gr"]) if g["gr"] else 160

        vp1, vr1 = gv(t1)
        vp2, vr2 = gv(t2)
        vm1 = rec[t1]["vs"].get(ven, {"m":0})["m"]
        vm2 = rec[t2]["vs"].get(ven, {"m":0})["m"]

        won = int(row["team1_won"])
        rows.append({
            "team1_won": won,
            "t1_enc": team_le.transform([t1])[0], "t2_enc": team_le.transform([t2])[0],
            "toss_enc": team_le.transform([row["toss_winner"]])[0],
            "toss_winner_bat_first": 1 if row["toss_winner"]==t1 else 0,
            "toss_field": 1 if row["toss_decision"]=="field" else 0,
            "venue_freq": vfreq.get(ven,0), "venue_avg_inn1": vavg.get(ven,160),
            "recent5_diff": w5_1-w5_2, "recent10_diff": w10_1-w10_2,
            "cum_pct_diff": c1-c2, "t1_h2h_pct": hp, "t1_h2h_n": len(hl),
            "home_diff": 0,
            "t1_matches": rec[t1]["m"], "t2_matches": rec[t2]["m"],
            "season_num": pd.to_numeric(str(row["season"]).replace("/","."),errors="coerce") or 2008,
            "elo_diff": rec[t1]["elo"]-rec[t2]["elo"],
            "venue_win_pct_diff": vp1-vp2,
            "team1_venue_win_pct": vp1, "team2_venue_win_pct": vp2,
            "team1_venue_matches": vm1, "team2_venue_matches": vm2,
            "team1_avg_runs_at_venue": vr1, "team2_avg_runs_at_venue": vr2,
        })

        # Update
        e1, e2 = rec[t1]["elo"], rec[t2]["elo"]
        exp = 1/(1+10**((e2-e1)/400))
        rec[t1]["elo"] += 32*(won-exp)
        rec[t2]["elo"] += 32*((1-won)-(1-exp))
        rec[t1]["w"]+=won; rec[t1]["m"]+=1; rec[t1]["r"].append(won); rec[t1]["gr"].append(row["inn1_runs"])
        rec[t2]["w"]+=1-won; rec[t2]["m"]+=1; rec[t2]["r"].append(1-won); rec[t2]["gr"].append(row["inn2_runs"])
        for team,runs,w in [(t1,row["inn1_runs"],won),(t2,row["inn2_runs"],1-won)]:
            if ven not in rec[team]["vs"]: rec[team]["vs"][ven]={"w":0,"m":0,"r":[]}
            rec[team]["vs"][ven]["m"]+=1; rec[team]["vs"][ven]["w"]+=w; rec[team]["vs"][ven]["r"].append(runs)
        if (t1,t2) not in h2h: h2h[(t1,t2)]=[]
        h2h[(t1,t2)].append(won)

    return pd.DataFrame(rows), team_le


def generate_match_probabilities(model, train_hist, schedule_df):
    """Generate pre-match probabilities for all schedule matches."""
    # Build feature state from training data replay
    hist = train_hist.copy()
    hist["batting_team"] = hist["batting_team"].apply(normalise)
    hist["bowling_team"] = hist["bowling_team"].apply(normalise)
    hist = hist.sort_values("date").reset_index(drop=True)

    all_teams = sorted(set(hist["batting_team"].unique()) | set(hist["bowling_team"].unique()))
    all_teams = sorted(set(all_teams) | set(schedule_df["team1"].unique()) | set(schedule_df["team2"].unique()))

    rec = {t: {"w":0,"m":0,"r":[],"elo":1500.0,"vs":{},"gr":[]} for t in all_teams}
    h2h = {}

    # Replay training history
    for _, row in hist.iterrows():
        t1, t2 = row["batting_team"], row["bowling_team"]
        ven = row["venue"]
        won = int(row["team1_won"])
        e1, e2 = rec[t1]["elo"], rec[t2]["elo"]
        exp = 1/(1+10**((e2-e1)/400))
        rec[t1]["elo"] += 32*(won-exp)
        rec[t2]["elo"] += 32*((1-won)-(1-exp))
        rec[t1]["w"]+=won; rec[t1]["m"]+=1; rec[t1]["r"].append(won); rec[t1]["gr"].append(row["inn1_runs"])
        rec[t2]["w"]+=1-won; rec[t2]["m"]+=1; rec[t2]["r"].append(1-won); rec[t2]["gr"].append(row["inn2_runs"])
        for team,runs,w in [(t1,row["inn1_runs"],won),(t2,row["inn2_runs"],1-won)]:
            if ven not in rec[team]["vs"]: rec[team]["vs"][ven]={"w":0,"m":0,"r":[]}
            rec[team]["vs"][ven]["m"]+=1; rec[team]["vs"][ven]["w"]+=w; rec[team]["vs"][ven]["r"].append(runs)
        if (t1,t2) not in h2h: h2h[(t1,t2)]=[]
        h2h[(t1,t2)].append(won)

    # Team encoder
    team_le = LabelEncoder()
    team_le.fit(all_teams)

    vfreq = pd.concat([hist["venue"], schedule_df["venue"]]).value_counts(normalize=True)
    vavg = hist.groupby("venue")["inn1_runs"].mean()

    # Generate features for each schedule match
    probs = []
    for _, row in schedule_df.iterrows():
        t1, t2, ven = row["team1"], row["team2"], row["venue"]
        r1, r2 = rec[t1]["r"], rec[t2]["r"]
        c1 = rec[t1]["w"]/max(rec[t1]["m"],1)
        c2 = rec[t2]["w"]/max(rec[t2]["m"],1)
        w5_1 = np.mean(r1[-5:]) if len(r1)>=5 else 0.5
        w5_2 = np.mean(r2[-5:]) if len(r2)>=5 else 0.5
        w10_1 = np.mean(r1[-10:]) if len(r1)>=10 else 0.5
        w10_2 = np.mean(r2[-10:]) if len(r2)>=10 else 0.5

        hl = h2h.get((t1,t2),[]) + h2h.get((t2,t1),[])
        hp = np.mean(hl) if hl else 0.5

        def gv(t):
            vs = rec[t]["vs"].get(ven, {"w":0,"m":0,"r":[]})
            if vs["m"]>=3: return vs["w"]/vs["m"], np.mean(vs["r"]) if vs["r"] else 160
            g = rec[t]
            return g["w"]/max(g["m"],1) if g["m"]>0 else 0.5, np.mean(g["gr"]) if g["gr"] else 160

        vp1, vr1 = gv(t1)
        vp2, vr2 = gv(t2)
        vm1 = rec[t1]["vs"].get(ven, {"m":0})["m"]
        vm2 = rec[t2]["vs"].get(ven, {"m":0})["m"]

        X = np.array([[
            team_le.transform([t1])[0], team_le.transform([t2])[0],
            team_le.transform([normalise(row["toss_winner"])])[0],
            1 if normalise(row["toss_winner"])==t1 else 0,
            1 if row["toss_decision"]=="field" else 0,
            vfreq.get(ven,0), vavg.get(ven,160),
            w5_1-w5_2, w10_1-w10_2, c1-c2, hp, len(hl), 0,
            rec[t1]["m"], rec[t2]["m"], 2026.0,
            rec[t1]["elo"]-rec[t2]["elo"],
            vp1-vp2, vp1, vp2, vm1, vm2, vr1, vr2,
        ]])

        proba = model.predict_proba(X)[0, 1]
        # Shrink toward 0.5
        proba_cal = 0.5 + (proba - 0.5) * 0.5
        probs.append(proba_cal)

    return np.array(probs)


def simulate_season_fast(base_probs, schedule_df, teams, n_sim=10000):
    """Fast Monte Carlo using pre-computed probabilities."""
    n_matches = len(schedule_df)

    # For each match in each simulation: draw outcome from base probability
    # No sequential state update — just independent draws
    np.random.seed(42)
    outcomes = np.random.random((n_sim, n_matches)) < base_probs[np.newaxis, :]  # (n_sim, n_matches)
    outcomes = outcomes.astype(int)

    # Compute points per simulation
    points = np.zeros((n_sim, len(teams)), dtype=int)
    wins = np.zeros((n_sim, len(teams)), dtype=int)
    team_idx = {t: i for i, t in enumerate(teams)}

    for mi in range(n_matches):
        row = schedule_df.iloc[mi]
        t1_idx = team_idx[row["team1"]]
        t2_idx = team_idx[row["team2"]]
        # outcomes[:, mi] = 1 means team1 won
        t1_wins = outcomes[:, mi]
        t2_wins = 1 - t1_wins
        points[:, t1_idx] += 2 * t1_wins
        points[:, t2_idx] += 2 * t2_wins
        wins[:, t1_idx] += t1_wins
        wins[:, t2_idx] += t2_wins

    return outcomes, points, wins


def compute_projections(outcomes, points, wins, schedule_df, teams, base_probs):
    """Compute all output projections."""
    n_sim = len(points)
    team_idx = {t: i for i, t in enumerate(teams)}

    # 1. Match predictions
    match_preds = []
    for mi in range(len(schedule_df)):
        row = schedule_df.iloc[mi]
        p = base_probs[mi]
        match_preds.append({
            "match_id": row["match_id"], "date": row["date"],
            "team1": row["team1"], "team2": row["team2"],
            "venue": row["venue"],
            "p_team1_win": round(p, 4), "p_team2_win": round(1-p, 4),
            "most_likely": row["team1"] if p >= 0.5 else row["team2"],
        })

    # 2-4. Standings / playoff / winner
    playoff_counts = {t: 0 for t in teams}
    winner_counts = {t: 0 for t in teams}
    final_counts = {t: 0 for t in teams}
    points_data = {t: [] for t in teams}

    for sim in range(n_sim):
        ranked = sorted(teams, key=lambda t: points[sim, team_idx[t]], reverse=True)
        for i, t in enumerate(ranked):
            points_data[t].append(int(points[sim, team_idx[t]]))
            if i == 0: winner_counts[t] += 1
            if i < 2: final_counts[t] += 1
            if i < 4: playoff_counts[t] += 1

    pts_table = []
    for rank, team in enumerate(teams, 1):
        pts = sorted(points_data[team], reverse=True)
        pts_table.append({
            "rank": rank, "team": team,
            "mean_points": round(np.mean(pts), 1),
            "p5_points": int(np.percentile(pts, 5)),
            "p50_points": int(np.percentile(pts, 50)),
            "p95_points": int(np.percentile(pts, 95)),
            "std_points": round(np.std(pts), 1),
            "playoff_probability": round(playoff_counts[team] / n_sim, 4),
        })

    playoff_odds = []
    for team in teams:
        playoff_odds.append({
            "team": team,
            "p_top1": round(winner_counts[team] / n_sim, 4),
            "p_top2": round(final_counts[team] / n_sim, 4),
            "p_top4": round(playoff_counts[team] / n_sim, 4),
        })
    playoff_odds.sort(key=lambda x: x["p_top4"], reverse=True)

    winner_odds = []
    for rank, (team, count) in enumerate(
        sorted(winner_counts.items(), key=lambda x: x[1], reverse=True), 1
    ):
        winner_odds.append({"rank": rank, "team": team, "p_winner": round(count / n_sim, 4)})

    return match_preds, pts_table, playoff_odds, winner_odds


def save_outputs(match_preds, pts_table, playoff_odds, winner_odds):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Add research notice as comment in each file
    for data, filename in [
        (match_preds, "match_predictions.csv"),
        (pts_table, "points_table_projection.csv"),
        (playoff_odds, "playoff_odds.csv"),
        (winner_odds, "winner_odds.csv"),
    ]:
        df = pd.DataFrame(data)
        path = OUTPUT_DIR / filename
        # Write notice as header comment
        with open(path, "w") as f:
            f.write(f"# {RESEARCH_NOTICE}\n")
        df.to_csv(path, mode="a", index=False)
        print(f"  Saved: output/{filename} ({len(data)} rows)")


def main():
    print("=" * 60)
    print("IPL 2026 Tournament Simulation")
    print("=" * 60)
    print(f"\n{RESEARCH_NOTICE}\n")
    t0 = time.time()

    # Load data
    print("[1/5] Loading data...")
    hist = pd.read_csv(DATA_PATH)
    train_hist = hist[~hist["season"].isin(["2025","2026"])].copy()
    schedule = pd.read_csv(SCHEDULE_PATH)
    teams = sorted(set(schedule["team1"].unique()) | set(schedule["team2"].unique()))
    print(f"  Train: {len(train_hist)} | Schedule: {len(schedule)} matches | Teams: {len(teams)}")

    # Train model
    print("\n[2/5] Training production model...")
    feat_df, team_le = build_train_features(train_hist)
    X_train = feat_df[FEATURE_COLS].values
    y_train = feat_df["team1_won"].values
    model = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print(f"  Trained on {len(X_train)} matches")

    # Generate probabilities
    print("\n[3/5] Generating match probabilities...")
    base_probs = generate_match_probabilities(model, train_hist, schedule)
    print(f"  Mean probability: {base_probs.mean():.4f}")
    print(f"  Range: [{base_probs.min():.4f}, {base_probs.max():.4f}]")
    print(f"  Std: {base_probs.std():.4f}")

    # Simulate
    print(f"\n[4/5] Running 10,000 Monte Carlo simulations...")
    outcomes, points, wins = simulate_season_fast(base_probs, schedule, teams, n_sim=10000)

    # Projections
    print("\n[5/5] Computing projections...")
    match_preds, pts_table, playoff_odds, winner_odds = compute_projections(
        outcomes, points, wins, schedule, teams, base_probs
    )
    save_outputs(match_preds, pts_table, playoff_odds, winner_odds)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Simulation complete in {elapsed:.1f}s")
    print(f"{'='*60}")
    print(f"\n{RESEARCH_NOTICE}\n")

    # Show results
    print("Sample Match Predictions (first 10):")
    print(f"  {'#':>3} {'Date':<12} {'Team 1':<25} {'Team 2':<25} {'P(T1)':>6} {'P(T2)':>6} {'Favorite':<25}")
    print(f"  {'-'*110}")
    for mp in match_preds[:10]:
        print(f"  {mp['match_id']:>3} {mp['date']:<12} {mp['team1']:<25} {mp['team2']:<25} "
              f"{mp['p_team1_win']:>6.1%} {mp['p_team2_win']:>6.1%} {mp['most_likely']:<25}")

    print("\nProjected Points Table:")
    print(f"  {'Rank':>4} {'Team':<30} {'Mean':>7} {'P5':>5} {'P50':>5} {'P95':>5} {'Std':>5} {'Playoff%':>9}")
    print(f"  {'-'*78}")
    for pt in pts_table:
        print(f"  {pt['rank']:>4} {pt['team']:<30} {pt['mean_points']:>7.1f} "
              f"{pt['p5_points']:>5} {pt['p50_points']:>5} {pt['p95_points']:>5} "
              f"{pt['std_points']:>5.1f} {pt['playoff_probability']:>9.1%}")

    print("\nPlayoff Odds:")
    print(f"  {'Team':<30} {'P(1st)':>7} {'P(Top2)':>8} {'P(Playoffs)':>11}")
    print(f"  {'-'*58}")
    for po in playoff_odds:
        print(f"  {po['team']:<30} {po['p_top1']:>7.1%} {po['p_top2']:>8.1%} {po['p_top4']:>11.1%}")

    print("\nTournament Winner Probabilities:")
    print(f"  {'Rank':>4} {'Team':<30} {'P(Winner)':>10}")
    print(f"  {'-'*46}")
    for wo in winner_odds:
        print(f"  {wo['rank']:>4} {wo['team']:<30} {wo['p_winner']:>10.1%}")

    print(f"\nAssumptions:")
    print(f"  Model: RandomForest (Elo+Venue, 24 features)")
    print(f"  Probabilities shrunk 50% toward 0.5 (calibration)")
    print(f"  Matches simulated independently (no sequential Elo update)")
    print(f"  League stage only (no playoffs)")
    print(f"  2 pts/win, 0 pts/loss")
    print(f"  Toss from schedule (not re-simulated)")
    print(f"  10,000 Monte Carlo iterations")
    print(f"  Runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()

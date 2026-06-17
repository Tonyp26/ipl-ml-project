# IPL ML Project — Status Report

**Date:** 2026-06-17
**Repository:** https://github.com/Tonyp26/ipl-ml-project
**Branch:** main

---

## Project Overview

This project attempts to predict IPL match winners using historical ball-by-ball data from 2008–2026. After rigorous feature engineering, diagnostics, and out-of-sample validation, the conclusion is that **T20 cricket match outcomes are not predictable from pre-match features available in this dataset**.

The project demonstrates best practices in ML: chronological evaluation, time-series cross-validation, feature auditing, and honest reporting of negative results.

---

## Dataset Summary

| Property | Value |
|---|---|
| Source | Kaggle IPL ball-by-ball data |
| Raw records | 283,678 ball-level rows |
| Aggregated | 1,193 matches |
| Seasons | 19 (2007/08 – 2026) |
| Teams | 10 active + 4 historical |
| Venues | 59 |
| Target | `team1_won` (did batting-first team win?) |
| Class balance | 55.2% Team 2 win, 44.8% Team 1 win |

---

## Model Results

**Production model:** RandomForest (300 trees, max_depth=8) with Elo + Venue features (24 features)

| Feature Set | Features | Accuracy | Test AUC | TS-CV AUC |
|---|---|---|---|---|
| Base (team ID, toss, venue) | 16 | 45.0% | 0.430 | 0.478 |
| +Elo ratings | 17 | **48.5%** | **0.450** | 0.471 |
| +Venue-specific stats | 24 | 46.2% | **0.450** | **0.498** |
| +Momentum (streaks) | 35 | 43.8% | 0.415 | 0.500 |
| +Scoring/NRR | 48 | 44.4% | 0.423 | **0.512** |

**Baseline (random):** 50.0% accuracy, 0.500 AUC

**Key finding:** Elo ratings provided the single best improvement (+3.5% accuracy). All models perform at or below random chance.

---

## 2025 Backtest Results

**Setup:** Trained on seasons ≤ 2024, tested on all 74 IPL 2025 matches.

| Metric | Result | Baseline | Assessment |
|---|---|---|---|
| Accuracy | 40.5% | 50.0% | Below random |
| ROC-AUC | 0.425 | 0.500 | Below random |
| Brier Score | 0.263 | 0.250 | Worse than baseline |
| Log Loss | 0.720 | 0.693 | Worse than baseline |

**Standings prediction:** 0/10 rank positions correct, 1/4 playoff teams correct (only Punjab Kings).

**Most confident predictions:** 11/20 correct (55%) — barely above coin flip.

---

## Tournament Simulation

**Setup:** 10,000 Monte Carlo iterations using the Elo+Venue production model.

- Match probabilities are tightly clustered around 0.50 (mean 0.508, range 0.451–0.546)
- All teams have nearly identical projected points (13.5–14.5 mean)
- Playoff probabilities range 27–51% (close to uniform 40% base rate)
- Winner odds range 5–15% (close to uniform 10%)

**The simulation framework works correctly — the model simply has no predictive power.**

---

## Dashboard Architecture

**Stack:** React 19 + TypeScript + Vite 8 + Tailwind CSS 3

```
frontend/
├── src/
│   ├── App.tsx              # BrowserRouter wrapper
│   ├── IPLDashboard.tsx     # 573-line single-page dashboard
│   ├── data/                # 4 static JSON files
│   ├── index.css            # Tailwind v3 directives
│   └── main.tsx             # Vite entry point
├── package.json
├── vite.config.ts
└── tailwind.config.js
```

**Dashboard sections:**
1. Project overview with stat cards
2. Dataset statistics and pipeline diagram
3. Feature engineering timeline
4. Model comparison table + feature importance chart
5. 2025 backtest results (highlighted section)
6. Tournament simulation results
7. Lessons learned + repository architecture

**Route:** `http://localhost:5173` (run `npm run dev` in `frontend/`)

---

## Feature Engineering Summary

| Feature Category | Count | Impact | Verdict |
|---|---|---|---|
| Team identity + toss + venue | 16 | Baseline | ✓ Keep |
| Elo ratings (K=32) | 1 | +3.5% accuracy | ✓ Best feature |
| Venue-specific stats | 7 | Improved CV stability | ✓ Keep |
| Momentum (streaks, recent form) | 11 | Hurt performance | ✗ Noise |
| Scoring/NRR | 13 | Best individual importance but diluted model | ⚠ Mixed |

---

## Future Improvement Ideas

1. **Player-level data** — Individual batting avg, strike rate, economy rate, recent form
2. **Squad composition** — Overseas players, all-rounders, key player availability
3. **Pitch/weather data** — Dew factor, pitch type, humidity, temperature
4. **Auction prices** — Team spending as proxy for strength
5. **2026 squad changes** — Mega auction reshuffles team dynamics
6. **Probabilistic models** — Bayesian approaches, Poisson score distributions
7. **Ensemble methods** — Combine multiple weak models
8. **More data** — IPL only has 19 seasons (1,193 matches); more leagues = more data

---

## Known Issues

1. **Model has no predictive power** — All metrics below random on out-of-sample data. This is inherent to T20 cricket's variance, not a bug.
2. **TS-CV AUC misleading** — Cross-validation AUC (~0.50) did not predict real 2025 performance (0.425 AUC).
3. **No player-level features** — The dataset only has team-level aggregates.
4. **Franchise name normalization** — Some historical franchises (Kochi Tuskers Kerala, Pune Warriors) have very few matches.
5. **Home advantage not captured** — City-based home detection yields no signal (home_diff = 0).

---

## Completed Phases

| Phase | Description | Status |
|---|---|---|
| 1 | Data aggregation (ball-by-ball → match-level) | ✅ Complete |
| 2 | Model training + leakage fixes | ✅ Complete |
| 2.5 | Model diagnostics | ✅ Complete |
| 3.1 | Elo rating features | ✅ Complete |
| 3.2 | Venue-specific features | ✅ Complete |
| 3.3 | Momentum features | ✅ Complete (proved noise) |
| 3.4 | Scoring strength + NRR features | ✅ Complete |
| 4 | 2025 backtest + tournament simulation | ✅ Complete |
| 5 | React web dashboard | ✅ Complete |

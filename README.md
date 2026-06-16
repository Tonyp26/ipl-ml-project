# IPL Match Winner Prediction

> Machine learning project predicting IPL match winners using historical data (2008–2026).

## Project Overview

This project builds a classifier to predict whether the team batting first wins an IPL match, using **only pre-match features** — information available before the first ball is bowled. The goal is to simulate real-world match prediction, not post-hoc analysis.

After rigorous diagnostics and iterative feature engineering, the key finding is that **T20 cricket match outcomes are nearly unpredictable from pre-match features alone**. The best model achieves ~46% accuracy and ~0.50 AUC on time-aware cross-validation — essentially at random chance.

## Dataset

| Source | Format | Records |
|---|---|---|
| Kaggle IPL ball-by-ball (2008–2026) | `IPL.csv` | 283,678 ball-level records |
| Aggregated match data | `data/processed/matches.csv` | 1,193 matches |
| Seasons covered | 2007/08 – 2026 | 19 seasons |
| Teams | 10 active + historical | Including franchise name changes |

The raw data is ball-by-ball and is aggregated to match-level summaries through the pipeline.

## Data Pipeline

### 1. Ball-by-Ball → Match Aggregation

```
notebooks/01_aggregate_matches.py
```

Aggregates 283K ball-level records into 1,193 match-level summaries:
- Innings scores, wickets, balls faced
- Powerplay (overs 1–6) and death over (16+) runs
- Run rates per phase
- Rolling team strength (win %, recent form)
- Head-to-head records
- Venue statistics

### 2. Feature Engineering

All features are computed **chronologically** — for each match, only data from **prior matches** is used. No future information leaks into the feature set.

| Feature Category | Count | Examples |
|---|---|---|
| Team identity | 3 | Encoded team IDs, toss winner |
| Toss | 2 | Toss decision, bat-first choice |
| Venue | 2 | Venue frequency, avg score |
| Team strength | 3 | Cumulative win % diff, experience |
| Recent form | 2 | Last 5/10 win % diff |
| Head-to-head | 2 | Historical win %, match count |
| Elo rating | 1 | Elo difference (K=32) |
| Venue-specific | 7 | Team win % at venue, avg runs |
| Momentum | 11 | Streaks, last 3/5 win % |

### 3. Data Normalization

Franchise name changes are mapped to canonical names:

| Original | Canonical |
|---|---|
| Delhi Daredevils | Delhi Capitals |
| Kings XI Punjab | Punjab Kings |
| Royal Challengers Bangalore | Royal Challengers Bengaluru |
| Deccan Chargers | Sunrisers Hyderabad |
| Rising Pune Supergiant | Rising Pune Supergiants |

## Model Development

### Train/Test Split

**Chronological split** — trains on earlier seasons, tests on later seasons (last 3 seasons as test set). This prevents temporal data leakage where future matches inform training.

### Cross-Validation

**Time-Series Split (5-fold expanding window)** — each fold trains on earlier data and validates on later data, respecting temporal ordering.

### Key Milestones

1. **Baseline Model** — RandomForest with random stratified split achieved 55.7% accuracy, 0.50 AUC (later found to have temporal leakage)
2. **Leakage Discovery** — Post-match innings features (runs, run rates) were being used to predict outcomes, creating trivial but useless predictions
3. **Chronological Split** — Replaced random split with season-based split; accuracy dropped from 55.7% to 45.0%, revealing true model performance
4. **Time-Aware CV** — Replaced random K-fold with TimeSeriesSplit; confirmed models perform near random
5. **Diagnostics** — Verified target encoding, feature computation, no future leakage, and model calibration
6. **Elo Features** — Added team Elo ratings; improved accuracy to 48.5%
7. **Venue Features** — Added venue-specific team performance; improved CV stability
8. **Momentum Features** — Added streaks and recent form; hurt performance (noise)

## Experiment Results

| Feature Set | Features | Accuracy | Test AUC | TS-CV AUC |
|---|---|---|---|---|
| Base | 16 | 45.0% | 0.430 | 0.478 |
| +Elo | 17 | **48.5%** | 0.450 | 0.471 |
| +Elo+Venue | 24 | 46.2% | **0.450** | **0.498** |
| +Elo+Venue+Momentum | 35 | 43.8% | 0.415 | 0.500 |

### Key Findings

- **T20 cricket is inherently unpredictable** from pre-match features. Even the best model performs at or below random chance.
- **Elo ratings** were the single best feature addition (+3.5% accuracy).
- **Venue-specific stats** improved cross-validation stability but not test accuracy.
- **Momentum/streak features** were noise and hurt performance.
- **Random splits inflate accuracy** through temporal leakage — always use chronological evaluation for time-series prediction.

## Repository Structure

```
├── notebooks/
│   ├── 01_aggregate_matches.py      # Ball-by-ball → match aggregation
│   ├── 02_train_model.py            # Model training pipeline
│   ├── 03_model_diagnostics.py      # Target, feature, and model audits
│   ├── 04_elo_features.py           # Elo rating calculation
│   ├── 05_venue_features.py         # Venue-specific team strength
│   ├── 06_momentum_features.py      # Form and streak features
│   └── 07_scoring_strength_features.py  # NRR and batting/bowling strength
├── data/
│   ├── IPL.csv                      # Raw ball-by-ball data (gitignored)
│   └── processed/
│       ├── matches.csv              # Aggregated match data (gitignored)
│       ├── matches_with_elo.csv     # + Elo features (gitignored)
│       ├── matches_with_venue.csv   # + Venue features (gitignored)
│       └── matches_with_momentum.csv # + Momentum features (gitignored)
├── models/                          # Generated artifacts (gitignored)
│   ├── random_forest_model.joblib
│   ├── team_encoder.joblib
│   ├── evaluation.json
│   └── *.png                        # Diagnostic plots
├── requirements.txt                 # System dependencies
├── requirements-ipl.txt             # IPL project dependencies
├── .gitignore
└── README.md
```

## Requirements

```
pandas >= 2.3.0
numpy >= 2.3.0
scikit-learn >= 1.7.0
matplotlib >= 3.10.0
seaborn >= 0.13.0
```

## Usage

```bash
# 1. Aggregate ball-by-ball data to match level
python notebooks/01_aggregate_matches.py

# 2. Generate feature sets
python notebooks/04_elo_features.py
python notebooks/05_venue_features.py
python notebooks/06_momentum_features.py

# 3. Train and evaluate model (auto-detects latest feature file)
python notebooks/02_train_model.py

# 4. Run diagnostics
python notebooks/03_model_diagnostics.py
```

## Future Work

- **Scoring strength & NRR features** — Batting vs bowling strength separation, net run rate as a predictive feature
- **Player-level features** — Individual player form, key player availability
- **Tournament simulation** — Simulate full 2026 IPL season to predict standings and winner
- **Model ensembling** — Combine multiple weak models for marginal gains
- **Web dashboard** — React-based visualization of predictions and model insights
- **External data** — Weather, pitch conditions, auction prices, squad changes

## License

MIT

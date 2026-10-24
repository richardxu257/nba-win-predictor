# Recorded model evaluation

Frozen fit and calibration use 2022–23 and 2023–24 only. Results below are rolling pregame forecasts; prior completed test games update features without refitting parameters.

Selected model: **logistic_regression**, followed by logistic probability calibration.

| Test season | Games | Accuracy | 95% day-bootstrap interval | Log loss | Brier score | Elo accuracy | Home-only accuracy |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 2024-25 | 1,321 | 67.68% | 65.07%–70.21% | 0.6039 | 0.2085 | 65.03% | 54.50% |
| 2025-26 | 1,322 | 67.40% | 64.92%–69.84% | 0.6056 | 0.2089 | 67.02% | 55.52% |

The intervals summarize within-season day-to-day sampling variation. They do not measure systematic data errors, future-season drift or unobserved injury information. Lower log loss/Brier is better. All-Star games are excluded; play-in and Cup games are included. Separate regular/cup, playoff and play-in statistics and calibration bins appear in `reports/metrics.json`.

The base model fits 2,436 games through March 2024. Probability calibration uses 203 later games in April–June 2024. Candidate validation log losses: logistic_regression: 0.6220, hist_gradient_boosting: 0.6407.

## Data gaps and interpretation

No sufficient historical pregame injury/lineup snapshots, effective-dated coaching assignments or true defender-player tracking were obtained. Coach features are inactive in this shipped model. Player co-appearance statistics are game-level production proxies. Current roster membership can be fetched at prediction time; injuries and expected minutes still need verification. See `DATA_SOURCES.md`.

## Largest standardized logistic coefficients

These describe the fitted model, not causal effects or isolated feature importance. Correlated features can distribute or reverse coefficients. The calibration layer subsequently transforms the raw probability.

| Feature | Standardized coefficient |
| --- | ---: |
| diff_elo | 0.4102 |
| diff_win_10 | -0.2548 |
| diff_off_rating_10 | 0.2065 |
| diff_pace_40 | -0.2064 |
| diff_margin_10 | 0.1637 |
| elo_probability | 0.1530 |
| diff_player_vs_team | 0.1488 |
| diff_b2b | -0.1224 |
| diff_player_trend | -0.1210 |
| diff_off_rating_40 | -0.1172 |
| diff_pace_10 | 0.1112 |
| diff_tov_rate_40 | -0.0769 |

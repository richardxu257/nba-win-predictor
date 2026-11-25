# 2026–27 NBA season backtest

Scoreboard snapshot: 2026-11-24T21:50:38-06:00 (Central time). Evaluated 260 completed games from 2026-10-20 through 2026-11-24.

The frozen logistic regression model uses the original 2022–24 fit and calibration. Each forecast uses box scores from earlier game dates and estimated rotations.

| Model | Accuracy | Log loss | Brier score | ROC AUC |
| --- | ---: | ---: | ---: | ---: |
| NBA win predictor | 66.92% | 0.6012 | 0.2074 | 0.7392 |
| Elo | 67.69% | 0.6124 | 0.2125 | 0.7309 |
| Home team baseline | 51.54% | 0.6973 | 0.2521 | 0.5000 |

The model picked 174 of 260 winners. The 95% accuracy interval from resampling game dates was 60.80%–72.73%.

## Home win probability calibration

| Probability bin | Games | Mean probability | Actual home win rate |
| --- | ---: | ---: | ---: |
| 10%–20% | 5 | 18.4% | 20.0% |
| 20%–30% | 22 | 24.7% | 13.6% |
| 30%–40% | 39 | 35.1% | 20.5% |
| 40%–50% | 42 | 45.4% | 52.4% |
| 50%–60% | 37 | 55.6% | 59.5% |
| 60%–70% | 54 | 65.5% | 59.3% |
| 70%–80% | 33 | 74.4% | 66.7% |
| 80%–90% | 27 | 84.1% | 85.2% |
| 90%–100% | 1 | 91.8% | 100.0% |

## Coverage

Verified game IDs and final scores against ESPN’s 36 daily scoreboards. The SportsDataverse archives supplied box scores for 258 evaluated games. ESPN final scores supplied labels for 2 additional games on the snapshot date. Those games used the same prior-day features as the other games on that date.

Games still live or pending at the snapshot: Dallas Mavericks at Denver Nuggets.

[Game predictions](2026-27_predictions_through_2026-11-24.csv) · [Full metrics and source hashes](2026-27_metrics_through_2026-11-24.json)

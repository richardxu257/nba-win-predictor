# NBA win predictor

Predict an NBA game's winner and both teams' win probabilities using only information available before the game. This repository contains a runnable Python package, trained model, season holdout predictions, metrics, data provenance and tests.

## Quick start

Python 3.11+ (3.12 used for the recorded run):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e . --no-deps
nba-predict download --years 2023 2024 2025 2026 2027
nba-predict train --output artifacts
nba-predict predict --home BOS --away NY --date 2026-10-24 --model artifacts/model.joblib
```

For new dates replace the example date with the game date. Team abbreviations and player/team IDs are ESPN identifiers (e.g. `NY`, not `NYK`). Run `nba-predict download --years 2027 --refresh` before current-season forecasts. Download years are ending years. The archive can lag completed games; prediction JSON includes latest history and staleness in days. A refresh failure raises an error rather than silently manufacturing data.

The checked-in `models/model.joblib` is the frozen 2022–24 model. Omit `--model` to use it. Only load trusted model artifacts; joblib deserialization runs Python object-loading code. Rebuild from source to avoid loading a serialized model.

To forecast every scheduled game with one history build:

```bash
nba-slate --date 2026-10-24 --live-rosters
```

The slate command uses ESPN’s scoreboard for schedule fields only. It uses the shipped coach-free model; use the single-game CLI for custom coaching inputs.

### Current rosters and availability

Default rotations are estimates from each team's previous five games, excluding players subsequently observed playing for another team. `--live-rosters` uses today's ESPN membership and prior player minutes instead; this flag is rejected for historical dates. It is not an injury feed:

```bash
nba-predict predict --home BOS --away NY --date 2026-10-24 --live-rosters
```

For actual pregame availability, supply a JSON object mapping ESPN team IDs to player IDs and expected minutes, and use `--rosters path/to/rotation.json`. Each included team must total 200–300 minutes; zero minutes exclude an unavailable player. Omitted teams use the default estimator. Supply all expected participants for an overridden team. IDs and minute allocations are validated; players without observed history receive neutral player form.

Conceptual schema (replace these placeholders with real IDs/minutes):

```json
{"ESPN_TEAM_ID": {"ESPN_PLAYER_ID": 32.0, "ANOTHER_PLAYER_ID": 28.0}}
```

Probabilities reflect data and model uncertainty, not certainty about injuries or lineups. A probability of 0.65 means a model estimate of a 65% chance, not a guarantee that 65% of similar forecasts are correct; inspect held-out calibration bins.

## Model and features

Two fixed candidates are compared on a chronological validation window: regularized logistic regression and histogram gradient-boosted decision trees. The better validation log loss wins. A logistic (Platt) calibration layer is fit on a later, disjoint window. See [MODEL_CARD.md](MODEL_CARD.md) for measured results and `reports/metrics.json` for the full metrics. A neural network is unnecessary for this initial few-thousand-game tabular dataset.

Features include:

- Elo strength with home advantage and offseason regression; rolling 10/40-game wins, scoring margin and head-to-head results.
- Estimated offensive/defensive rating, effective FG%, turnover rate, offensive rebounding share, free-throw rate and pace.
- Rest, back-to-back status and games in the prior four days.
- Prior estimated rotation, minute-weighted player production, 10-versus-40-game player trends, stale-history decay, player-versus-team history and shrunk player co-appearance performance differences.
- Player history coverage and rotation coverage to expose cold starts.
- Optional coach win rates and coach-versus-team rates from verified effective-dated assignments. These are uninformative in the shipped model because no sufficient historical coaching dataset was obtained.

The player production score is a documented box-score heuristic in `features.py`, not RAPM or an official player rating. Increasing/decreasing production changes player features over time without assuming every player follows the same aging curve. Sparse matchup observations shrink toward neutral.

## Evaluation protocol

1. History begins at the 2022–23 season; no pre-2022 data enters feature state.
2. Candidate fitting: through January 31, 2024. Candidate selection: February–March 2024 by log loss.
3. Refit selected candidate: through March 31, 2024. Probability calibration: April–June 2024. No calibration game is used to fit the underlying candidate model.
4. Freeze parameters and evaluate 2024–25 and 2025–26 independently in the report. Later completed games update rolling feature state, including 2024–25 results for 2025–26 forecasts; parameters and calibration remain frozen.
5. Predict every game on a date before ingesting any results from that date. Actual target-game roster, minutes, box scores and season-end averages never enter its features. No shuffled train/test split.

This is a **rolling one-day-ahead backtest**, not a preseason prediction of an entire season. Results include regular season, playoffs and play-in games, with separate breakdowns. ESPN type 5 marks play-ins. All-Star exhibitions are explicitly excluded by restricting teams to the 30 NBA franchise IDs, because ESPN labels exhibitions as type 2. Cup games, including the Cup final, follow provider classifications; a full regular/cup group may therefore contain 1,231 games rather than 1,230. Current 2026–27 games are excluded from training, candidate selection, calibration and reported tests.

Report accuracy, ROC-AUC, log loss, Brier score, calibration bins, day-block bootstrap accuracy intervals, a training-frequency home baseline and an Elo baseline. Confidence intervals reflect sampling variability within the observed season, not all sources of model uncertainty. Held-out results are not used to select another model after the fact. A stronger baseline can outperform this model; the report retains that comparison.

## Files

- `nba_predictor/`: downloader, input validation, chronological feature state, model, prediction CLI, optional current roster reader.
- `models/`: trained frozen model.
- `reports/`: measured metrics, coverage audit, all test-game predictions, source hashes, example current forecast.
- `tests/`: future/target-game leakage, same-day isolation, trade assignment, probability metrics and coach effective dates.
- `DATA_SOURCES.md`: exact providers, URLs and material data gaps.

## Coaching extension

Create a CSV with `team_id,coach_id,effective_date` using ESPN team IDs and a stable coach identifier. `effective_date` must be the first date the assignment was publicly in effect; include opening assignments and all changes. Then run:

```bash
nba-predict --coaches data/coach_assignments.csv train --output artifacts_with_coaches
nba-predict --coaches data/coach_assignments.csv predict --home BOS --away NY --date 2026-10-24 --model artifacts_with_coaches/model.joblib
```

Use the same verified historical file for training and inference, extended with newer known assignments. Do not substitute a season-end coach list. The model cannot learn meaningful coach effects unless dated assignments cover training history.

## Development

```bash
pip install -e '.[dev]'
pytest -q
```

GitHub Actions runs the small deterministic tests without downloading the full archives. Raw data and intermediate feature matrices are not committed. Recorded metrics and downloaded file hashes are committed; upstream archives may be corrected over time, so keep the original local cache to reproduce an exact snapshot.

# Data provenance and availability

All results in this project use publicly downloadable **ESPN team and player game box scores**, processed and published by SportsDataverse/hoopR. No paid service or API key is required.

| Dataset | Upstream / distribution | Use |
| --- | --- | --- |
| Team game boxes | [SportsDataverse ESPN NBA team boxes](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_nba_team_boxscores) | Results, home advantage, form, possessions/efficiency estimates, rest, head-to-head, Elo |
| Player game boxes | [SportsDataverse ESPN NBA player boxes](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_nba_player_boxscores) | Prior rotations, player form/trend, player versus team, player co-appearances |
| Current schedule | ESPN `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD` | Daily slate; only schedule fields used |
| Processing code | [hoopR NBA data](https://github.com/sportsdataverse/hoopR-nba-data) | Source transformations and schema context |
| Optional current membership | ESPN `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{ESPN_TEAM_ID}/roster` | `predict --live-rosters`, captured at request time, today only |

Files use the season **ending year**: `team_box_2023.parquet` means 2022–23. Identifiers throughout this project are **ESPN IDs**, not NBA Stats IDs. Downloaded files and manifest hashes are stored locally; the checked-in `reports/data_manifest.json` records the exact files used in the reported run. The upstream release assets can be revised, so future refreshed runs may differ. `observed_at` records this project's retrieval/inspection time, not when ESPN first published a box score.

## What is unavailable or insufficient

- **True individual defensive matchups** (defender assignments, possessions defended, shot difficulty): not present in these box scores. The implemented co-appearance feature compares players' prior game-level production when they both played. It does **not** establish they defended each other or shared the court at the same moment. Pair histories are shrunk toward zero when sparse.
- **Historical pregame injuries, announced starters and complete time-stamped transaction/availability snapshots**: not obtained. Historical features use prior five team games plus observed player assignments. This can miss recent trades, new signings, returns and late scratches. Current ESPN membership can be fetched at inference; it does not certify availability. A supplied expected-minutes rotation overrides the estimate.
- **Reliable effective-dated coaching assignments**: not obtained. I inspected the public [NBA Stats coaches archive](https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/nba_stats_coaches). Despite its release description saying per-game, the inspected 2023 CSV has season/team records and no game ID or effective date. It is unsafe to treat these as pregame historical assignments. Coach win rate and coach-versus-team win rate are implemented behind `--coaches`; they are constant/uninformative in the shipped model. Supplying a verified historical assignment file requires retraining before those features become useful.
- **Travel distance, time-zone changes, altitude, betting lines, tracking/lineup adjusted plus-minus**: not used. Rest and congestion are included. Possession counts are box-score estimates and do not explicitly adjust overtime pace.
- **Player growth/decline**: captured by 10-game versus 40-game production trends and elapsed-time decay. This is a statistical proxy, not a causal estimate of aging, development, or injury recovery. Rookies have a cold-start limitation.

Data rights remain with their respective owners/providers. Public accessibility does not transfer ownership or provide a commercial redistribution license. Raw archives are excluded from Git; obtain them from the linked provider. Model-generated predictions, metrics and retrieval hashes are included for auditability.

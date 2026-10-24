"""Download versioned public ESPN box-score archives maintained by SportsDataverse."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import requests

BASE = 'https://github.com/sportsdataverse/sportsdataverse-data/releases/download'


def download(root, years, refresh=False):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/'manifest.json').read_text()) if (root/'manifest.json').exists() else {}
    for year in years:
        for kind in ('team', 'player'):
            name = f'{kind}_box_{year}.parquet'
            path = root/name
            url = f'{BASE}/espn_nba_{kind}_boxscores/{name}'
            if refresh or not path.exists():
                response = requests.get(url, timeout=120)
                response.raise_for_status()
                temp = path.with_suffix('.tmp')
                temp.write_bytes(response.content)
                pd.read_parquet(temp)  # Validate before replacing a good cached file.
                temp.replace(path)
            manifest[name] = dict(url=url, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                  observed_at=datetime.now(timezone.utc).isoformat())
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def load(root):
    root = Path(root)
    frames = []
    players = []
    coverage = []
    for path in sorted(root.glob('team_box_*.parquet')):
        year = int(path.stem.split('_')[-1])
        if year < 2023: continue
        ppath = root/f'player_box_{year}.parquet'
        if not ppath.exists(): raise ValueError(f'Missing player archive: {ppath}')
        t = pd.read_parquet(path); p = pd.read_parquet(ppath)
        for d in (t,p):
            d['date'] = pd.to_datetime(d.game_date, utc=True).dt.tz_localize(None).dt.normalize()
        # ESPN: 1 preseason, 2 regular season/cup/all-star, 3 playoffs, 5 play-in.
        # All-Star exhibitions share type 2: explicitly restrict to 30 NBA franchises.
        excluded_exhibitions = int((~t.team_id.between(1,30)).sum()/2)
        t = t[t.season_type.isin([2,3,5]) & t.team_id.between(1,30) & t.opponent_team_id.between(1,30) & t.team_score.notna() & t.opponent_team_score.notna()].copy()
        if t.duplicated(['game_id','team_id']).any(): raise ValueError('Duplicate team/game rows')
        valid = []
        for gid, g in t.groupby('game_id'):
            if len(g) != 2 or set(g.team_home_away) != {'home','away'}:
                raise ValueError(f'Incomplete game {gid}')
            h = g[g.team_home_away == 'home'].iloc[0]; a = g[g.team_home_away == 'away'].iloc[0]
            if h.team_score == a.team_score: continue  # Unplayed placeholder, never a loss.
            if h.opponent_team_score != a.team_score or a.opponent_team_score != h.team_score:
                raise ValueError(f'Inconsistent scores: {gid}')
            valid.append(gid)
        t = t[t.game_id.isin(valid)]
        p = p[p.game_id.isin(valid)].copy()
        if p.duplicated(['game_id','team_id','athlete_id']).any(): raise ValueError('Duplicate player rows')
        required = set(zip(t.game_id,t.team_id))
        observed = set(zip(p.loc[p.minutes.fillna(0)>0,'game_id'],p.loc[p.minutes.fillna(0)>0,'team_id']))
        if required-observed: raise ValueError(f'Missing player box scores: {len(required-observed)} team-games')
        t['season_end'] = year; p['season_end'] = year
        coverage.append(dict(season=f'{year-1}-{str(year)[-2:]}', games=len(valid),
                             first=str(t.date.min().date()), last=str(t.date.max().date()),
                             player_rows=len(p), regular_games=int((t.season_type==2).sum()/2),
                             postseason_games=int((t.season_type==3).sum()/2),
                             play_in_games=int((t.season_type==5).sum()/2),
                             excluded_all_star_exhibitions=excluded_exhibitions))
        frames.append(t); players.append(p)
    if not frames: raise ValueError('No archives; run download first')
    teams = pd.concat(frames, ignore_index=True)
    if teams.duplicated(['game_id','team_id']).any(): raise ValueError('Cross-season duplicate games')
    return teams, pd.concat(players, ignore_index=True), coverage

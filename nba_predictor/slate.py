"""One feature-state build for a full daily slate."""
import argparse
from datetime import datetime
import json
from zoneinfo import ZoneInfo
import joblib
import pandas as pd
import requests
from .data import load
from .features import build
from .live import rotation_snapshot
from .model import probability


def main():
    parser=argparse.ArgumentParser(description='Forecast an ESPN daily NBA slate')
    parser.add_argument('--date',default=datetime.now(ZoneInfo('America/New_York')).date().isoformat())
    parser.add_argument('--data',default='data/raw')
    parser.add_argument('--model',default='models/model.joblib')
    parser.add_argument('--live-rosters',action='store_true')
    args=parser.parse_args()
    date=pd.Timestamp(args.date).normalize()
    bundle=joblib.load(args.model)
    if bundle.get('coaches_enabled',False): raise ValueError('Use the single-game CLI with the coaching file for this model')
    if date<=pd.Timestamp(bundle['trained_through']): raise ValueError('Date must follow training period')
    if args.live_rosters and date.date()!=datetime.now(ZoneInfo('America/New_York')).date():
        raise ValueError('Live roster membership is only valid for today')
    teams,players,_=load(args.data)
    teams=teams[teams.date<date]; players=players[players.date<date]
    _,state=build(teams,players,emit_features=False)
    url=f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date:%Y%m%d}&limit=100'
    response=requests.get(url,timeout=30); response.raise_for_status()
    results=[]
    for event in response.json().get('events',[]):
        competition=event['competitions'][0]
        competitors={c['homeAway']:c for c in competition['competitors']}
        home,away=competitors['home'],competitors['away']
        h,a=int(home['team']['id']),int(away['team']['id'])
        rotations=None; snapshot=None
        if args.live_rosters: rotations,snapshot=rotation_snapshot([h,a],state)
        f=state.features(h,a,date,rotations)
        p=float(probability(bundle,pd.DataFrame([f]))[0])
        results.append(dict(game_id=event['id'],tipoff_utc=event['date'],home=home['team']['abbreviation'],
                            away=away['team']['abbreviation'],predicted_winner=(home if p>=.5 else away)['team']['displayName'],
                            home_win_probability=p,away_win_probability=1-p,winner_probability=max(p,1-p),
                            roster_snapshot=snapshot))
    print(json.dumps(dict(as_of_date=str(date.date()),schedule_source=url,model=bundle['model_name'],
                          latest_history=str(teams.date.max().date()),games=results,
                          notes=['Frozen model trained/calibrated through 2023-24.',
                                 'No live injury clearance; rotations are estimates.',
                                 'Historical coach assignments and true defensive matchup tracking unavailable.']),indent=2))

if __name__=='__main__': main()

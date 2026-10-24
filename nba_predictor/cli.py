import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from .live import rotation_snapshot
import json
from pathlib import Path
import joblib
import pandas as pd
from .data import download,load
from .features import build
from .model import train,probability


def coaches(path):
    if not path: return None
    frame=pd.read_csv(path)
    required={'team_id','coach_id','effective_date'}
    if not required.issubset(frame): raise ValueError(f'Coaches must contain {required}')
    frame['effective_date']=pd.to_datetime(frame.effective_date)
    if frame.duplicated(['team_id','effective_date']).any(): raise ValueError('Ambiguous coach assignments')
    return frame


def main():
    parser=argparse.ArgumentParser(description='NBA pregame win probabilities')
    parser.add_argument('--data',default='data/raw')
    parser.add_argument('--coaches',help='Optional effective-dated coach assignments CSV')
    sub=parser.add_subparsers(dest='command',required=True)
    d=sub.add_parser('download'); d.add_argument('--years',type=int,nargs='+',default=[2023,2024,2025,2026,2027]); d.add_argument('--refresh',action='store_true')
    t=sub.add_parser('train'); t.add_argument('--output',default='artifacts')
    p=sub.add_parser('predict'); p.add_argument('--home',required=True,help='ESPN team abbreviation, e.g. BOS'); p.add_argument('--away',required=True)
    p.add_argument('--date',required=True,help='YYYY-MM-DD, as-of start of day')
    p.add_argument('--live-rosters',action='store_true',help='Fetch current ESPN membership for today only; not injury clearance')
    p.add_argument('--model',default='models/model.joblib'); p.add_argument('--rosters',help='JSON: ESPN team ID -> player ID -> expected minutes; pregame inputs only')
    args=parser.parse_args()
    if args.command=='download':
        download(args.data,args.years,args.refresh); return
    teams,players,coverage=load(args.data)
    coachdata=coaches(args.coaches)
    if args.command=='train':
        # Current-season data must not affect training or evaluation, even via preprocessing.
        frame,_=build(teams[teams.season_end<=2026],players[players.season_end<=2026],coachdata)
        _,report=train(frame,args.output)
        output=Path(args.output)
        (output/'coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
        frame.to_parquet(output/'features.parquet',index=False)
        print(json.dumps({k:{'accuracy':v['model']['accuracy'],'log_loss':v['model']['log_loss'],'brier':v['model']['brier']} for k,v in report['seasons'].items()},indent=2))
        return
    date=pd.Timestamp(args.date).normalize()
    if date.tzinfo is not None: raise ValueError('Use a plain YYYY-MM-DD date')
    bundle=joblib.load(args.model)  # Load trusted local model files only.
    if bool(args.coaches)!=bundle.get('coaches_enabled',False):
        raise ValueError('Use the coaching configuration used to train this model')
    if date<=pd.Timestamp(bundle['trained_through']): raise ValueError('Prediction date must follow model training period')
    historical=teams[teams.date<date]; priorplayers=players[players.date<date]
    _,state=build(historical,priorplayers,coachdata,emit_features=False)
    lookup={str(r.team_abbreviation).upper():int(r.team_id) for r in historical.itertuples()}
    home,away=lookup[args.home.upper()],lookup[args.away.upper()]
    if home==away: raise ValueError('Teams must differ')
    rosters=None; snapshot=None
    if args.live_rosters:
        if args.rosters: raise ValueError('Choose either live membership or manual rotations')
        if date.date()!=datetime.now(ZoneInfo('America/New_York')).date():
            raise ValueError('Current roster snapshots can only be used for today; never backtest with them')
        rosters,snapshot=rotation_snapshot([home,away],state)
    if args.rosters:
        raw=json.loads(Path(args.rosters).read_text())
        rosters={int(t):{int(p):float(m) for p,m in roster.items()} for t,roster in raw.items()}
        for roster in rosters.values():
            if any(m<0 or m>60 for m in roster.values()) or not 200<=sum(roster.values())<=300:
                raise ValueError('Expected rotation minutes must total 200–300, each between 0 and 60')
    features=state.features(home,away,date,rosters)
    prob=float(probability(bundle,pd.DataFrame([features]))[0])
    latest=historical.date.max()
    print(json.dumps(dict(date=str(date.date()),home=args.home.upper(),away=args.away.upper(),
                         predicted_winner=args.home.upper() if prob>=.5 else args.away.upper(),
                         winner_probability=max(prob,1-prob),home_win_probability=prob,away_win_probability=1-prob,
                         model=bundle['model_name'],latest_history=str(latest.date()),history_age_days=(date-latest).days,
                         rotation_source='current ESPN membership and estimated minutes' if snapshot else ('pregame override' if rosters else 'estimated from prior five team games'),
                         roster_snapshot=snapshot,
                         notes=['No live injury feed; verify availability and expected minutes.',
                                'Coach features unavailable unless effective-dated assignments provided.',
                                'Co-appearance is not defender-versus-player tracking.']),indent=2))

if __name__=='__main__': main()

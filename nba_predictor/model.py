import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

META = {'game_id','date','season_end','season_type','home','away','home_name','away_name','target'}


def metrics(y,p):
    p = np.clip(np.asarray(p),1e-6,1-1e-6); y = np.asarray(y)
    bins = []
    for low in np.arange(0,1,.1):
        m = (p>=low)&(p<low+.1)
        if m.any(): bins.append(dict(low=round(float(low),1),count=int(m.sum()),mean_probability=float(p[m].mean()),observed_home_win_rate=float(y[m].mean())))
    return dict(games=len(y),accuracy=float(accuracy_score(y,p>=.5)),log_loss=float(log_loss(y,p,labels=[0,1])),
                brier=float(brier_score_loss(y,p)),auc=float(roc_auc_score(y,p)) if len(set(y))>1 else None,
                calibration=bins)


def logits(p):
    p=np.clip(p,1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)


def probability(bundle, frame):
    p = bundle['base'].predict_proba(frame[bundle['features']])[:,1]
    return bundle['calibrator'].predict_proba(logits(p))[:,1]


def train(frame, out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    cols = sorted(set(frame)-META)
    # Selection, refit, and probability calibration all remain in requested training seasons.
    fit = (frame.season_end<=2024)&(frame.date < '2024-02-01')
    val = (frame.date>='2024-02-01')&(frame.date<'2024-04-01')
    refit = (frame.season_end<=2024)&(frame.date<'2024-04-01')
    cal = (frame.season_end<=2024)&(frame.date>='2024-04-01')
    if min(fit.sum(),val.sum(),cal.sum())<50: raise ValueError('Insufficient training/validation/calibration games')
    candidates = {
        'logistic_regression':make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),LogisticRegression(C=.1,max_iter=3000)),
        'hist_gradient_boosting':HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,learning_rate=.05,l2_regularization=10,early_stopping=False,random_state=42)
    }
    scores={}
    for name, model in candidates.items():
        model.fit(frame.loc[fit,cols],frame.loc[fit,'target'])
        scores[name]=float(log_loss(frame.loc[val,'target'],model.predict_proba(frame.loc[val,cols])[:,1]))
    name=min(scores,key=scores.get); model=candidates[name]
    model.fit(frame.loc[refit,cols],frame.loc[refit,'target'])
    calibrator=LogisticRegression(C=1,max_iter=1000)
    calibrator.fit(logits(model.predict_proba(frame.loc[cal,cols])[:,1]),frame.loc[cal,'target'])
    bundle=dict(base=model,calibrator=calibrator,features=cols,model_name=name,trained_through='2024-06-30',coaches_enabled=bool(frame.diff_coach_known.ne(0).any() or frame.diff_coach_winrate.ne(0).any()))
    joblib.dump(bundle,out/'model.joblib')
    report=dict(model=name,validation_log_loss=scores,fit_games=int(refit.sum()),calibration_games=int(cal.sum()),
                protocol='Frozen model; rolling features updated after each completed game day; no holdout refitting.',seasons={})
    baseline_p = float(frame.loc[refit,'target'].mean())
    predictions=[]
    for year in (2025,2026):
        test=frame[frame.season_end==year].copy()
        if test.empty: raise ValueError(f'Missing requested test season {year}')
        p=probability(bundle,test)
        season=f'{year-1}-{str(year)[-2:]}'
        report['seasons'][season]=dict(model=metrics(test.target,p),elo=metrics(test.target,test.elo_probability),
                                     home_baseline=metrics(test.target,np.full(len(test),baseline_p)),
                                     regular_season=metrics(test.loc[test.season_type==2,'target'],p[test.season_type==2]),
                                     postseason=metrics(test.loc[test.season_type==3,'target'],p[test.season_type==3]),
                                     play_in=metrics(test.loc[test.season_type==5,'target'],p[test.season_type==5]))
        # Paired day-block bootstrap gives uncertainty without assuming games on one day independent.
        rng=np.random.default_rng(42)
        day=pd.DataFrame({'date':test.date,'correct':(p>=.5)==test.target}).groupby('date').correct.agg(['sum','count'])
        acc=[]
        for _ in range(1000):
            sampled=day.iloc[rng.integers(0,len(day),len(day))]
            acc.append(sampled['sum'].sum()/sampled['count'].sum())
        report['seasons'][season]['accuracy_95pct_day_bootstrap']=np.quantile(acc,[.025,.975]).tolist()
        pred=test[sorted(META)].copy()
        pred['home_win_probability']=p; pred['away_win_probability']=1-p
        pred['predicted_winner']=np.where(p>=.5,test.home_name,test.away_name)
        pred['winner_probability']=np.maximum(p,1-p)
        predictions.append(pred)
    pd.concat(predictions).sort_values(['date','game_id']).to_csv(out/'test_predictions.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    return bundle,report

"""Exercise the full fitting protocol with tiny estimators to audit date boundaries."""
import numpy as np
import pandas as pd
from nba_predictor import model


def test_training_and_calibration_never_fit_test_years(monkeypatch,tmp_path):
    fits=[]
    class Estimator:
        def fit(self,x,y):
            fits.append(np.asarray(x).copy())
            return self
        def predict_proba(self,x):
            return np.tile([.45,.55],(len(x),1))
    monkeypatch.setattr(model,'make_pipeline',lambda *args:Estimator())
    monkeypatch.setattr(model,'HistGradientBoostingClassifier',lambda **kwargs:Estimator())
    monkeypatch.setattr(model,'LogisticRegression',lambda **kwargs:Estimator())
    monkeypatch.setattr(model.joblib,'dump',lambda *args:None)
    dates=list(pd.date_range('2023-01-01',periods=70))+list(pd.date_range('2024-02-01',periods=55))+list(pd.date_range('2024-04-01',periods=55))
    for year in (2025,2026): dates+=list(pd.date_range(f'{year}-01-01',periods=10))
    frame=pd.DataFrame({'date':dates})
    frame['season_end']=frame.date.dt.year
    frame['game_id']=range(len(frame)); frame['target']=np.arange(len(frame))%2
    frame['season_type']=2
    for year in (2025,2026):
        idx=frame.index[frame.season_end==year]
        frame.loc[idx[-2:],'season_type']=3
        frame.loc[idx[-3:-2],'season_type']=5
    for name in ('home','away','home_name','away_name'): frame[name]=1
    frame['diff_coach_known']=0.; frame['diff_coach_winrate']=0.
    frame['elo_probability']=.55
    # Sentinel feature identifies forbidden held-out rows in fit calls.
    frame['sentinel']=np.where(frame.season_end>=2025,999999.,0.)
    _,report=model.train(frame,tmp_path)
    assert all(not np.any(values==999999.) for values in fits)
    assert set(report['seasons'])=={'2024-25','2025-26'}
    assert report['calibration_games']==55

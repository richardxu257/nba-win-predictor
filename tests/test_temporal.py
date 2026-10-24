import copy
import pandas as pd
import pytest
from nba_predictor.features import State,build
from nba_predictor.model import metrics


def fixture():
    teams=[]; players=[]
    for day in (1,2,3):
        for team,opp,score,other,loc in ((1,2,110,100,'home'),(2,1,100,110,'away')):
            common=dict(game_id=day,date=pd.Timestamp(f'2023-01-0{day}'),season_end=2023,season_type=2,team_id=team,
                        team_abbreviation=str(team),opponent_team_id=opp,team_home_away=loc,team_score=score,opponent_team_score=other,
                        field_goals_attempted=90,field_goals_made=40,offensive_rebounds=10,defensive_rebounds=30,turnovers=12,
                        free_throws_attempted=20,three_point_field_goals_made=10)
            teams.append(common)
            players.append(dict(game_id=day,team_id=team,athlete_id=team,opponent_team_id=opp,date=common['date'],minutes=30.,points=20.,
                                rebounds=5.,assists=5.,steals=1.,blocks=1.,turnovers=2.,field_goals_attempted=15.,field_goals_made=8.,
                                free_throws_attempted=5.,free_throws_made=4.))
    return pd.DataFrame(teams),pd.DataFrame(players)


def test_target_game_and_future_never_change_pregame_features():
    t,p=fixture(); original,_=build(t,p)
    altered=t.copy(); altered.loc[altered.game_id>=2,'team_score']+=40
    changed=p.copy(); changed.loc[changed.game_id>=2,'points']+=70
    new,_=build(altered,changed)
    cols=[c for c in original if c not in ('target',)]
    pd.testing.assert_frame_equal(original.loc[original.game_id<=2,cols],new.loc[new.game_id<=2,cols])


def test_same_day_results_not_available():
    t,p=fixture(); t.loc[t.game_id==2,'date']=pd.Timestamp('2023-01-01'); p.loc[p.game_id==2,'date']=pd.Timestamp('2023-01-01')
    frame,_=build(t,p)
    assert frame.iloc[0].elo_probability == frame.iloc[1].elo_probability


def test_player_assignment_removes_traded_player():
    s=State(); s.rotation[1]=[{9:30}]; s.assignment[9]=2
    assert s.roster(1)=={}


def test_probabilistic_metrics_penalize_confident_errors():
    assert metrics([0,1],[.99,.01])['log_loss'] > metrics([0,1],[.5,.5])['log_loss']


def test_coach_effective_date():
    c=pd.DataFrame([dict(team_id=1,coach_id='A',effective_date=pd.Timestamp('2023-01-02'))])
    s=State(c)
    assert s.coach(1,pd.Timestamp('2023-01-01')) is None
    assert s.coach(1,pd.Timestamp('2023-01-02')) == 'A'


def test_archive_excludes_allstar_and_includes_playin(tmp_path):
    from nba_predictor.data import load
    t,p=fixture()
    t['game_date']=t.date; p['game_date']=p.date
    t.loc[t.game_id==3,'season_type']=5
    extra=t[t.game_id==1].copy(); extra['game_id']=4
    extra['team_id']+=30; extra['opponent_team_id']+=30
    t=pd.concat([t,extra],ignore_index=True)
    t.to_parquet(tmp_path/'team_box_2023.parquet')
    p.to_parquet(tmp_path/'player_box_2023.parquet')
    teams,_,coverage=load(tmp_path)
    assert set(teams.game_id)=={1,2,3}
    assert coverage[0]['play_in_games']==1
    assert coverage[0]['excluded_all_star_exhibitions']==1


def test_archive_rejects_inconsistent_score(tmp_path):
    from nba_predictor.data import load
    t,p=fixture(); t['game_date']=t.date; p['game_date']=p.date
    t.loc[0,'team_score']=999
    t.to_parquet(tmp_path/'team_box_2023.parquet'); p.to_parquet(tmp_path/'player_box_2023.parquet')
    with pytest.raises(ValueError,match='Inconsistent scores'):
        load(tmp_path)


def test_state_only_replay_matches_backtest_state():
    t,p=fixture()
    _,full=build(t,p)
    frame,fast=build(t,p,emit_features=False)
    assert frame.empty
    date=pd.Timestamp('2023-01-04')
    assert full.features(1,2,date)==fast.features(1,2,date)

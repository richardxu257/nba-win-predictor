"""State machine: predict an entire day, then ingest its completed games.

No target-game lineups, minutes, scores, or season-end statistics enter features.
"""
from collections import defaultdict
from functools import lru_cache
import sys
import numpy as np
import pandas as pd


@lru_cache(maxsize=64)
def _weights(length):
    weights = np.exp(np.linspace(-2,0,length))
    return weights/weights.sum()


def weighted(values, n):
    if not values: return 0.
    a = np.asarray(values[-n:], dtype=float)
    return float(np.dot(a,_weights(len(a))))


class State:
    def __init__(self, coaches=None):
        self.teams = defaultdict(list)
        self.players = defaultdict(list)
        self.rotation = defaultdict(list)
        self.assignment = {}
        self.pairs = defaultdict(list)
        self.h2h = defaultdict(list)
        self.elo = defaultdict(lambda: 1500.)
        self.coaches = coaches if coaches is not None else pd.DataFrame()
        self.coach_games = defaultdict(list)
        self.last_season = None

    def coach(self, team, date):
        if self.coaches.empty: return None
        c = self.coaches[(self.coaches.team_id == team) & (self.coaches.effective_date <= date)]
        return None if c.empty else str(c.sort_values('effective_date').iloc[-1].coach_id)

    def roster(self, team):
        # Estimate rotation using previous five games only; remove known traded players.
        recent = self.rotation[team][-5:]
        totals = defaultdict(float)
        for rotation in recent:
            for pid, mins in rotation.items():
                if self.assignment.get(pid) == team: totals[pid] += mins/max(len(recent),1)
        return totals

    def team_features(self, team, opponent, date, roster=None):
        history = self.teams[team]
        result = {'elo':self.elo[team], 'games_observed':min(len(history),82)}
        for metric in ('win','margin','off_rating','def_rating','pace','efg','tov_rate','orb','ft_rate'):
            vals = [r[metric] for r in history[-40:]]
            for n in (10,40): result[f'{metric}_{n}'] = weighted(vals,n)
        rest = (date-history[-1]['date']).days if history else 7
        result['rest'] = min(rest,7)
        result['b2b'] = float(rest == 1)
        result['games_last4days'] = sum(0 < (date-r['date']).days <= 4 for r in history[-4:])
        result['h2h_margin'] = weighted(self.h2h[(team,opponent)],8)
        rotation = self.roster(team) if roster is None else roster
        result['rotation_size'] = len(rotation)
        result['rotation_minutes'] = sum(rotation.values())
        result['player_form'] = result['player_trend'] = result['player_vs_team'] = 0.
        result['player_history_coverage'] = 0.
        denom = max(sum(rotation.values()),1)
        for pid, mins in rotation.items():
            records = self.players[pid]
            w = mins/denom
            if records:
                # Recency decay toward zero for stale/unobserved players; no fixed aging assumption.
                decay = np.exp(-max((date-records[-1]['date']).days-30,0)/180)
                values = [r['impact36'] for r in records[-40:]]
                result['player_form'] += w*weighted(values,10)*decay
                result['player_trend'] += w*(weighted(values,10)-weighted(values,40))*decay
                vs = [r['impact36'] for r in records if r['opponent']==opponent]
                result['player_vs_team'] += w*weighted(vs,10)*len(vs)/(len(vs)+10)
                result['player_history_coverage'] += w*min(len(records)/10,1)
        coach = self.coach(team,date)
        games = self.coach_games[coach] if coach else []
        vs = [win for opp,win in games if opp==opponent]
        result['coach_known'] = float(coach is not None)
        result['coach_winrate'] = (sum(w for _,w in games)+5)/(len(games)+10)
        result['coach_vs_team'] = (sum(vs)+5)/(len(vs)+10)
        return result

    def features(self, home, away, date, rosters=None):
        rosters = rosters or {}
        rh = rosters.get(home,self.roster(home)); ra = rosters.get(away,self.roster(away))
        h = self.team_features(home,away,date,rh); a = self.team_features(away,home,date,ra)
        f = {f'diff_{k}':h[k]-a[k] for k in h}
        f['elo_probability'] = 1/(1+10**((self.elo[away]-self.elo[home]-65)/400))
        f['min_player_coverage'] = min(h['player_history_coverage'],a['player_history_coverage'])
        f['min_rotation_minutes'] = min(h['rotation_minutes'],a['rotation_minutes'])
        value = support = 0.
        total = max(sum(rh.values())*sum(ra.values()),1)
        for p,pm in rh.items():
            for q,qm in ra.items():
                history = self.pairs[(p,q)]
                w = pm*qm/total
                value += w*weighted(history,10)*len(history)/(len(history)+10)
                support += w*min(len(history)/10,1)
        # This is prior game co-appearance performance, NOT defensive tracking.
        f['coappearance_impact_difference'] = value
        f['coappearance_support'] = support
        return f

    def update(self, game, boxes):
        home = game[game.team_home_away=='home'].iloc[0]
        away = game[game.team_home_away=='away'].iloc[0]
        h,a = int(home.team_id),int(away.team_id)
        win = float(home.team_score > away.team_score)
        expected = 1/(1+10**((self.elo[a]-self.elo[h]-65)/400))
        change = 20*(win-expected)
        self.elo[h] += change; self.elo[a] -= change
        for row, opp in ((home,away),(away,home)):
            team = int(row.team_id); opponent = int(opp.team_id)
            poss = max(float(row.field_goals_attempted-row.offensive_rebounds+row.turnovers+.44*row.free_throws_attempted),1)
            oppposs = max(float(opp.field_goals_attempted-opp.offensive_rebounds+opp.turnovers+.44*opp.free_throws_attempted),1)
            pace = (poss+oppposs)/2
            r = dict(date=row.date,win=float(row.team_score>opp.team_score),margin=float(row.team_score-opp.team_score),
                     off_rating=100*row.team_score/pace,def_rating=100*opp.team_score/pace,pace=pace,
                     efg=(row.field_goals_made+.5*row.three_point_field_goals_made)/max(row.field_goals_attempted,1),
                     tov_rate=row.turnovers/poss,orb=row.offensive_rebounds/max(row.offensive_rebounds+opp.defensive_rebounds,1),
                     ft_rate=row.free_throws_attempted/max(row.field_goals_attempted,1))
            self.teams[team].append(r)
            self.h2h[(team,opponent)].append(r['margin'])
            coach = self.coach(team,row.date)
            if coach: self.coach_games[coach].append((opponent,r['win']))
        rotations = defaultdict(dict); impacts = {}
        for row in boxes.itertuples():
            if pd.isna(row.minutes) or row.minutes <= 0: continue
            pid,team = int(row.athlete_id),int(row.team_id)
            impact = (row.points + 1.2*row.rebounds + 1.5*row.assists + 3*(row.steals+row.blocks)
                      -row.turnovers-(row.field_goals_attempted-row.field_goals_made)
                      -.5*(row.free_throws_attempted-row.free_throws_made))*36/max(row.minutes,8)
            self.players[pid].append(dict(date=row.date,impact36=float(impact),minutes=float(row.minutes),opponent=int(row.opponent_team_id)))
            self.assignment[pid] = team
            rotations[team][pid] = float(row.minutes); impacts[pid] = float(impact)
        for team in (h,a): self.rotation[team].append(rotations[team])
        for p in rotations[h]:
            for q in rotations[a]:
                d = impacts[p]-impacts[q]
                self.pairs[(p,q)].append(d); self.pairs[(q,p)].append(-d)


def build(teams, players, coaches=None, emit_features=True):
    state = State(coaches)
    boxes = {gid:g for gid,g in players.groupby('game_id')}
    records = []
    for date, daily in teams.groupby('date',sort=True):
        games = list(daily.groupby('game_id',sort=True))
        season = int(daily.season_end.iloc[0])
        if season != state.last_season:
            print(f'Building pregame features for season ending {season}',file=sys.stderr,flush=True)
        if state.last_season is not None and season != state.last_season:
            for team in state.elo: state.elo[team] = 1500+.75*(state.elo[team]-1500)
        state.last_season = season
        if emit_features:
            for gid,game in games:
                home = game[game.team_home_away=='home'].iloc[0]; away = game[game.team_home_away=='away'].iloc[0]
                records.append(dict(game_id=gid,date=date,season_end=season,season_type=int(home.season_type),
                                    home=int(home.team_id),away=int(away.team_id),home_name=home.team_abbreviation,
                                    away_name=away.team_abbreviation,target=int(home.team_score>away.team_score),
                                    **state.features(int(home.team_id),int(away.team_id),date)))
        for gid,game in games: state.update(game,boxes[gid])
    return pd.DataFrame(records),state

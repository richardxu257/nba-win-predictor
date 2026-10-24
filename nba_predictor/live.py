"""Current ESPN membership snapshots; never used for historical evaluation."""
from datetime import datetime, timezone
import requests


def rotation_snapshot(team_ids, state):
    rotations={}; sources=[]; memberships={}
    for team in team_ids:
        url=f'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team}/roster'
        response=requests.get(url,timeout=30); response.raise_for_status()
        payload=response.json(); athletes=payload.get('athletes',[])
        if not athletes: raise ValueError(f'No roster returned for ESPN team {team}')
        memberships[team]=[dict(player_id=int(a['id']),name=a.get('displayName','')) for a in athletes]
        rotation={}
        for athlete in athletes:
            pid=int(athlete['id']); records=state.players[pid]
            # The current roster identifies membership, not starting status or injury clearance.
            if records: rotation[pid]=sum(r['minutes'] for r in records[-5:])/min(len(records),5)
        if not rotation: raise ValueError('No prior minute history; supply an explicit pregame rotation')
        factor=240/sum(rotation.values())
        rotations[team]={pid:minutes*factor for pid,minutes in rotation.items()}
        sources.append(url)
    return rotations,dict(captured_at=datetime.now(timezone.utc).isoformat(),sources=sources,memberships=memberships,expected_minutes=rotations,
                          warning='Membership only; injuries and expected minutes are unverified. New players without history receive zero minutes.')

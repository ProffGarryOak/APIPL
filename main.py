# Extended FastAPI IPL Backend with Fantasy, Live Sim, Progress, Notifications, and AI Summary

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime
from fastapi import HTTPException
import math
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path
def deep_clean(data):
    if isinstance(data, dict):
        return {k: deep_clean(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [deep_clean(v) for v in data]
    elif isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return None
    elif isinstance(data, (np.generic,)):
        return data.item()
    elif isinstance(data, (pd.Timestamp, datetime)):
        return data.isoformat()
    else:
        return data


app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = Path("static/index.html")
    if html_path.exists():
        return html_path.read_text()
    return "<h1>Oops! index.html not found 😵‍💫</h1>"

# CORS setup for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data
matches = pd.read_csv('matches.csv')
matches['date'] = pd.to_datetime(matches['date'], errors='coerce', dayfirst=True)
matches['season'] = matches['date'].dt.year

deliveries = pd.read_csv('deliveries.csv')
orange_cap = pd.read_csv('orange_cap.csv')
purple_cap = pd.read_csv('purple_cap.csv')

@app.get("/api/runs_by_season")
def runs_by_season(season: Optional[int] = None, team: Optional[str] = None):
    df = matches.copy()
    if season:
        df = df[df['season'] == season]
    if team:
        df = df[(df['team1'] == team) | (df['team2'] == team)]

    match_ids = df['match_id'].astype(str)

    # Aggregate runs by season (and team if provided)
    deliveries_filtered = deliveries[deliveries['match_no'].astype(str).isin(match_ids)]
    if team:
        # Only runs scored by the team as batting_team
        deliveries_filtered = deliveries_filtered[deliveries_filtered['batting_team'] == team]

    runs_by_season = (
        df[['match_id', 'season']]
        .merge(
            deliveries_filtered.groupby('match_no')['runs_of_bat'].sum().reset_index(),
            left_on='match_id', right_on='match_no', how='left'
        )
        .groupby('season')['runs_of_bat'].sum()
        .reset_index()
        .rename(columns={'runs_of_bat': 'runs'})
    )

    # Fill NaN with 0 and convert to int
    runs_by_season['runs'] = runs_by_season['runs'].fillna(0).astype(int)
    runs_by_season['season'] = runs_by_season['season'].astype(int)

    return [
        {"season": int(row["season"]), "runs": int(row["runs"])}
        for _, row in runs_by_season.iterrows()
    ]
    df = matches.copy()
    if season:
        df = df[df['season'] == season]
    if team:
        df = df[(df['team1'] == team) | (df['team2'] == team)]

    total_matches = len(df)
    match_ids = df['match_id'].astype(str)

    total_runs = deliveries[deliveries['match_no'].astype(str).isin(match_ids)]['runs_of_bat'].sum()
    total_wickets = deliveries[(deliveries['match_no'].astype(str).isin(match_ids)) &
                               (deliveries['wicket_type'].notna())].shape[0]

    team_performance = {}
    if team:
        wins = df[df['match_winner'] == team].shape[0]
        losses = df[(df['match_winner'] != team) & (df['match_result'] == 'completed')].shape[0]
        draws = df[df['match_result'] == 'tied'].shape[0]
        no_results = df[df['match_result'] == 'no result'].shape[0]
        team_performance = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "no_results": no_results,
            "win_percentage": round(wins / (wins + losses) * 100, 2) if (wins + losses) > 0 else 0
        }

    result = matches.copy()
    result['season'] = result['season'].astype(int)
    result['runs'] = result['runs'].fillna(0).astype(int)
    
    return [
        {"season": int(row["season"]), "runs": int(row["runs"])}
        for _, row in result.replace({np.nan: None}).iterrows()
    ]

@app.get("/api/fantasy_points")
def fantasy_points(match_id: str):
    df = deliveries[deliveries['match_no'].astype(str) == match_id]
    points = {}
    for _, row in df.iterrows():
        # Batting
        batter = row['striker']
        points.setdefault(batter, 0)
        points[batter] += row['runs_of_bat']
        if row['runs_of_bat'] == 4:
            points[batter] += 1
        elif row['runs_of_bat'] == 6:
            points[batter] += 2
        # Bowling
        if pd.notna(row['wicket_type']):
            bowler = row['bowler']
            points.setdefault(bowler, 0)
            points[bowler] += 25
    sorted_points = sorted(points.items(), key=lambda x: x[1], reverse=True)
    return [{"player": p, "points": pts} for p, pts in sorted_points]

@app.get("/api/live_simulation")
def live_simulation(match_id: str, upto_ball: Optional[float] = None):
    df = deliveries[deliveries['match_no'].astype(str) == match_id].copy()
    if upto_ball:
        df['ball_no'] = df['over'] + 0.01  # mock up decimal overs
        df = df[df['ball_no'] <= upto_ball]
    total_runs = df['runs_of_bat'].sum()
    wickets = df['wicket_type'].notna().sum()
    return {
        "match_id": match_id,
        "balls": df.shape[0],
        "runs": int(total_runs),
        "wickets": int(wickets),
    }

@app.get("/api/match_progress")
def match_progress(match_id: str, upto_over: float):
    df = deliveries[deliveries['match_no'].astype(str) == match_id]
    df = df[df['over'] <= upto_over]
    runs = df['runs_of_bat'].sum()
    wickets = df['wicket_type'].notna().sum()
    return {
        "runs": int(runs),
        "wickets": int(wickets),
        "overs": upto_over,
    }

@app.get("/api/notifications")
def notifications(match_id: str):
    df = deliveries[deliveries['match_no'].astype(str) == match_id]
    events = []
    batter_runs = {}
    bowler_wickets = {}
    for _, row in df.iterrows():
        b = row['striker']
        batter_runs[b] = batter_runs.get(b, 0) + row['runs_of_bat']
        if batter_runs[b] == 50:
            events.append(f"Milestone: {b} scored 50 runs!")
        if pd.notna(row['wicket_type']):
            bowler = row['bowler']
            bowler_wickets[bowler] = bowler_wickets.get(bowler, 0) + 1
            if bowler_wickets[bowler] == 3:
                events.append(f"Milestone: {bowler} took 3 wickets!")
    return {"events": events}

@app.get("/api/ai_summary")
def ai_summary(match_id: str):
    match = matches[matches['match_id'].astype(str) == match_id].iloc[0].to_dict()
    summary = (
        f"In the {match['stage']} match at {match['venue']}, {match['team1']} faced off against {match['team2']} on {match['date'].date()}. "
        f"{match['team1']} scored {match['first_ings_score']}/{match['first_ings_wkts']} in the first innings, "
        f"while {match['team2']} chased with {match['second_ings_score']}/{match['second_ings_wkts']}. "
        f"The match was {match['match_result']} and {match['match_winner']} emerged victorious. "
        f"Top performer: {match['player_of_the_match']} - {match['top_scorer']} scored {match['highscore']}, "
        f"{match['best_bowling']} returned with {match['best_bowling_figure']}."
    )
    return {"summary": summary}
@app.get("/api/summary")
def get_summary(season: Optional[int] = None, team: Optional[str] = None):
    df = matches.copy()
    if season:
        df = df[df['season'] == season]
    if team:
        df = df[(df['team1'] == team) | (df['team2'] == team)]

    total_matches = len(df)
    match_ids = df['match_id'].astype(str)

    total_runs = deliveries[deliveries['match_no'].astype(str).isin(match_ids)]['runs_of_bat'].sum()
    total_wickets = deliveries[(deliveries['match_no'].astype(str).isin(match_ids)) &
                               (deliveries['wicket_type'].notna())].shape[0]

    team_performance = {}
    if team:
        wins = df[df['match_winner'] == team].shape[0]
        losses = df[(df['match_winner'] != team) & (df['match_result'] == 'completed')].shape[0]
        draws = df[df['match_result'] == 'tied'].shape[0]
        no_results = df[df['match_result'] == 'no result'].shape[0]
        team_performance = {
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "no_results": no_results,
            "win_percentage": round(wins / (wins + losses) * 100, 2) if (wins + losses) > 0 else 0
        }

    return {
        "total_matches": total_matches,
        "total_runs": int(total_runs),
        "total_wickets": int(total_wickets),
        "team_performance": team_performance if team else None
    }

@app.get("/api/top_players")
def get_top_players(season: Optional[int] = None, limit: int = 5):
    print("hii")
    if season:
        match_ids = matches[matches['season'] == season]['match_id'].astype(str)
        season_deliveries = deliveries[deliveries['match_no'].astype(str).isin(match_ids)]
    else:
        season_deliveries = deliveries.copy()

    top_batsmen = (season_deliveries.groupby('striker')
                   .agg(runs=('runs_of_bat', 'sum'),
                        balls=('striker', 'count'),
                        fours=('runs_of_bat', lambda x: (x == 4).sum()),
                        sixes=('runs_of_bat', lambda x: (x == 6).sum()))
                   .reset_index()
                   .sort_values('runs', ascending=False)
                   .head(limit))
    top_batsmen['strike_rate'] = round((top_batsmen['runs'] / top_batsmen['balls']) * 100, 2)

    top_bowlers = (season_deliveries[season_deliveries['wicket_type'].notna()]
                   .groupby('bowler')
                   .agg(wickets=('bowler', 'count'),
                        runs_conceded=('runs_of_bat', 'sum'))
                   .reset_index()
                   .sort_values(['wickets', 'runs_conceded'], ascending=[False, True])
                   .head(limit))

    return {
        "top_batsmen": top_batsmen.to_dict(orient='records'),
        "top_bowlers": top_bowlers.to_dict(orient='records')
    }






@app.get("/api/match_details")
def get_match_details(match_id: str):
    match_row = matches[matches['match_id'].astype(str) == str(match_id)]
    if match_row.empty:
        raise HTTPException(status_code=404, detail="Match ID not found")

    try:
        
        match = match_row.iloc[0].to_dict()
        
        match_deliveries = deliveries[deliveries['match_no'].astype(str) == str(match_id)]

        batting_performance = []
        for team in [match.get('team1'), match.get('team2')]:
            if not team:
                continue

            team_deliveries = match_deliveries[match_deliveries['batting_team'] == team]
            if team_deliveries.empty:
                continue

            runs = team_deliveries['runs_of_bat'].sum()
            wickets = team_deliveries['wicket_type'].notna().sum()
            overs = team_deliveries['over'].max()
            if pd.isna(overs):
                overs = 0.0  # Replace NaN with default safe value

            batsmen = (team_deliveries.groupby('striker')
                       .agg(runs=('runs_of_bat', 'sum'),
                            balls=('striker', 'count'),
                            fours=('runs_of_bat', lambda x: (x == 4).sum()),
                            sixes=('runs_of_bat', lambda x: (x == 6).sum()))
                       .reset_index()
                       .sort_values('runs', ascending=False))
            batsmen['strike_rate'] = round((batsmen['runs'] / batsmen['balls']) * 100, 2)

            bowlers = (match_deliveries[match_deliveries['bowling_team'] == team]
                       .groupby('bowler')
                       .agg(wickets=('bowler', 'count'),
                            runs=('runs_of_bat', 'sum'),
                            extras=('extras', 'sum'))
                       .reset_index()
                       .sort_values('wickets', ascending=False))
            bowlers['total_runs'] = bowlers['runs'] + bowlers['extras']

            batting_performance.append({
                "team": team,
                "runs": int(runs),
                "wickets": int(wickets),
                "overs": float(overs),
                "batsmen": batsmen.to_dict(orient='records'),
                "bowlers": bowlers.to_dict(orient='records')
            })

        match['performance'] = batting_performance
        
       
        return deep_clean(match)


        

    except Exception as e:
        print(f"[ERROR] match_id={match_id}: {e}")
        raise HTTPException(status_code=500, detail="Something went wrong processing this match.")

@app.get("/api/teams")
def get_teams():
    teams = sorted(set(matches['team1']).union(set(matches['team2'])))
    return {"teams": list(teams)}

@app.get("/api/team_runs_by_match")
def team_runs_by_match(team: str, season: Optional[int] = None):
    df = matches.copy()
    if season:
        df = df[df['season'] == season]
    df = df[(df['team1'] == team) | (df['team2'] == team)]

    match_ids = df['match_id'].astype(str)
    team_deliveries = deliveries[deliveries['match_no'].astype(str).isin(match_ids) & (deliveries['batting_team'] == team)]

    runs_by_match = (team_deliveries.groupby('match_no')
                     .agg(runs=('runs_of_bat', 'sum'))
                     .reset_index()
                     .sort_values('match_no'))

    return runs_by_match.to_dict(orient="records")

@app.get("/api/team_runrate_by_match")
def team_runrate_by_match(team: str, season: Optional[int] = None):
    df = matches.copy()
    if season:
        df = df[df['season'] == season]
    df = df[(df['team1'] == team) | (df['team2'] == team)]

    match_ids = df['match_id'].astype(str)
    team_deliveries = deliveries[deliveries['match_no'].astype(str).isin(match_ids) & (deliveries['batting_team'] == team)]

    runrate_by_match = (
        team_deliveries.groupby('match_no')
        .agg(
            runs=('runs_of_bat', 'sum'),
            balls=('runs_of_bat', 'count')
        )
        .reset_index()
    )
    runrate_by_match['run_rate'] = round((runrate_by_match['runs'] / runrate_by_match['balls']) * 6, 2)

    return runrate_by_match.to_dict(orient="records")
@app.get("/api/team_wickets_by_match")
def team_wickets_by_match(team: str, season: Optional[int] = None):
    df = matches.copy()
    if season:
        df = df[df['season'] == season]
    df = df[(df['team1'] == team) | (df['team2'] == team)]

    match_ids = df['match_id'].astype(str)
    team_deliveries = deliveries[deliveries['match_no'].astype(str).isin(match_ids) & (deliveries['batting_team'] == team)]

    wickets_by_match = (
        team_deliveries.groupby('match_no')
        .agg(wickets=('wicket_type', lambda x: x.notna().sum()))
        .reset_index()
    )

    return wickets_by_match.to_dict(orient="records")



@app.get("/api/team_run_rate_trend")
def team_run_rate_trend(team: str, season: Optional[int] = None):
    df = matches.copy()
    if season:
        df = df[df['season'] == season]

    match_ids = df[(df['team1'] == team) | (df['team2'] == team)]['match_id'].astype(str)
    d = deliveries[(deliveries['match_no'].astype(str).isin(match_ids)) & 
                   (deliveries['batting_team'] == team)].copy()

    def get_phase(over):
        if over <= 6: return "Powerplay"
        elif over <= 15: return "Middle"
        else: return "Death"
    d['phase'] = d['over'].apply(get_phase)

    trend = d.groupby(['match_no', 'phase']).agg(runs=('runs_of_bat', 'sum'), balls=('phase', 'count')).reset_index()
    trend['run_rate'] = round((trend['runs'] / trend['balls']) * 6, 2)
    return trend.to_dict(orient='records')


@app.get("/api/hard_hitters")
def hard_hitters(min_strike_rate: float = 140.0, min_sixes: int = 10):
    bats = deliveries.groupby('striker').agg(
        runs=('runs_of_bat', 'sum'),
        balls=('striker', 'count'),
        sixes=('runs_of_bat', lambda x: (x == 6).sum())
    ).reset_index()
    bats['strike_rate'] = round((bats['runs'] / bats['balls']) * 100, 2)
    result = bats[(bats['strike_rate'] >= min_strike_rate) & (bats['sixes'] >= min_sixes)]
    return result.sort_values(by='sixes', ascending=False).to_dict(orient='records')

@app.get("/api/clutch_performers_teamwise")
def clutch_performers_teamwise():
    merged = deliveries.merge(
        matches[['match_id', 'team1', 'team2']],
        left_on='match_no', right_on='match_id', how='left'
    )

    merged['innings_type'] = merged.apply(
        lambda row: 'chase' if row['batting_team'] == row['team2'] else 'defend', axis=1
    )

    # Batting: players who scored >= 30 in chase
    clutch_bat = merged[merged['innings_type'] == 'chase'].groupby(['match_no', 'batting_team', 'striker'])['runs_of_bat'].sum().reset_index()
    clutch_bat = clutch_bat[clutch_bat['runs_of_bat'] >= 30]

    # Bowling: players who took >= 2 wickets while defending
    clutch_bowl = merged[(merged['innings_type'] == 'defend') & (merged['wicket_type'].notna())]
    clutch_bowl = clutch_bowl.groupby(['match_no', 'bowling_team', 'bowler']).size().reset_index(name='wickets')
    clutch_bowl = clutch_bowl[clutch_bowl['wickets'] >= 2]

    result = {}

    for _, row in clutch_bat.iterrows():
        team = row['batting_team']
        result.setdefault(team, set()).add(row['striker'])

    for _, row in clutch_bowl.iterrows():
        team = row['bowling_team']
        result.setdefault(team, set()).add(row['bowler'])

    # Convert sets to sorted lists
    result = {team: sorted(list(players)) for team, players in result.items()}
    return {"clutch_performers": result}

@app.get("/api/consistent_performers_teamwise")
def consistent_performers_teamwise():
    merged = deliveries.merge(
        matches[['match_id', 'team1', 'team2']],
        left_on='match_no', right_on='match_id', how='left'
    )

    player_team_map = merged.groupby('striker')['batting_team'].agg(lambda x: x.mode()[0])

    bat_scores = merged.groupby(['match_no', 'striker'])['runs_of_bat'].sum().reset_index()
    bat_stats = bat_scores.groupby('striker')['runs_of_bat'].agg(['mean', 'std', 'count']).reset_index()
    bat_stats = bat_stats[(bat_stats['count'] >= 5) & (bat_stats['std'] <= 15)]

    bat_stats['team'] = bat_stats['striker'].map(player_team_map)

    result = {}
    for _, row in bat_stats.iterrows():
        team = row['team']
        result.setdefault(team, []).append({
            "player": row['striker'],
            "avg_runs": round(row['mean'], 2),
            "std_dev": round(row['std'], 2),
            "matches": int(row['count'])
        })

    return {"consistent_performers": result}

@app.get("/api/player_form")
def player_form(player: str):
    recent_matches = matches.sort_values("date", ascending=False).head(5)['match_id'].astype(str)
    recent_deliveries = deliveries[deliveries['match_no'].astype(str).isin(recent_matches)]

    bat = recent_deliveries[recent_deliveries['striker'] == player]
    bowl = recent_deliveries[recent_deliveries['bowler'] == player]

    return {
        "batting": {
            "runs": int(bat['runs_of_bat'].sum()),
            "balls": bat.shape[0],
            "strike_rate": round((bat['runs_of_bat'].sum() / bat.shape[0]) * 100, 2) if bat.shape[0] else 0,
            "fours": int((bat['runs_of_bat'] == 4).sum()),
            "sixes": int((bat['runs_of_bat'] == 6).sum()),
        },
        "bowling": {
            "wickets": int(bowl['wicket_type'].notna().sum()),
            "runs_conceded": int(bowl['runs_of_bat'].sum() + bowl['extras'].sum()),
            "economy": round(((bowl['runs_of_bat'].sum() + bowl['extras'].sum()) / bowl.shape[0]) * 6, 2) if bowl.shape[0] else 0,
        }
    }


@app.get("/api/compare_players")
def compare_players(player1: str, player2: str):
    def get_stats(player):
        bat = deliveries[deliveries['striker'] == player]
        bowl = deliveries[deliveries['bowler'] == player]
        return {
            "batting": {
                "runs": int(bat['runs_of_bat'].sum()),
                "balls": bat.shape[0],
                "strike_rate": round((bat['runs_of_bat'].sum() / bat.shape[0]) * 100, 2) if bat.shape[0] else 0
            },
            "bowling": {
                "wickets": int(bowl['wicket_type'].notna().sum()),
                "balls": bowl.shape[0],
                "economy": round(((bowl['runs_of_bat'].sum() + bowl['extras'].sum()) / bowl.shape[0]) * 6, 2) if bowl.shape[0] else 0
            }
        }

    return {
        player1: get_stats(player1),
        player2: get_stats(player2)
    }
@app.get("/api/head_to_head_player")
def head_to_head_player(batter: str, bowler: str):
    h2h = deliveries[(deliveries['striker'] == batter) & (deliveries['bowler'] == bowler)]
    return {
        "runs": int(h2h['runs_of_bat'].sum()),
        "balls": h2h.shape[0],
        "fours": int((h2h['runs_of_bat'] == 4).sum()),
        "sixes": int((h2h['runs_of_bat'] == 6).sum()),
        "dismissals": int(h2h[h2h['player_dismissed'] == batter].shape[0]),
        "strike_rate": round((h2h['runs_of_bat'].sum() / h2h.shape[0]) * 100, 2) if h2h.shape[0] else 0
    }
    
@app.get("/api/head_to_head_player")
def head_to_head_player(batter: str, bowler: str):
    h2h = deliveries[(deliveries['striker'] == batter) & (deliveries['bowler'] == bowler)]
    return {
        "runs": int(h2h['runs_of_bat'].sum()),
        "balls": h2h.shape[0],
        "fours": int((h2h['runs_of_bat'] == 4).sum()),
        "sixes": int((h2h['runs_of_bat'] == 6).sum()),
        "dismissals": int(h2h[h2h['player_dismissed'] == batter].shape[0]),
        "strike_rate": round((h2h['runs_of_bat'].sum() / h2h.shape[0]) * 100, 2) if h2h.shape[0] else 0
    }
    
@app.get("/api/team_insights")
def team_insights(team: str):
    team_deliv = deliveries[deliveries['batting_team'] == team]

    def phase(over):
        if over <= 6:
            return "Powerplay"
        elif over <= 15:
            return "Middle"
        else:
            return "Death"

    team_deliv['phase'] = team_deliv['over'].apply(phase)

    insights = team_deliv.groupby('phase').agg(
        runs=('runs_of_bat', 'sum'),
        balls=('over', 'count'),
        wickets=('wicket_type', lambda x: x.notna().sum())
    ).reset_index()

    insights['run_rate'] = round((insights['runs'] / insights['balls']) * 6, 2)

    return insights.to_dict(orient="records")
@app.get("/api/leaderboards")
def leaderboards():
    most_sixes = deliveries[deliveries['runs_of_bat'] == 6]['striker'].value_counts().head(5)
    best_economy = (deliveries[deliveries['wicket_type'].notna()]
                    .groupby('bowler')
                    .agg(runs=('runs_of_bat', 'sum'), balls=('bowler', 'count'))
                    .assign(economy=lambda x: round((x['runs'] / x['balls']) * 6, 2))
                    .sort_values('economy').head(5))

    best_finisers = (deliveries.groupby(['match_no', 'striker'])
                     .agg(runs=('runs_of_bat', 'sum'))
                     .reset_index()
                     .sort_values('match_no', ascending=False)
                     .groupby('striker').agg(total=('runs', 'sum'))
                     .sort_values('total', ascending=False).head(5))

    return {
        "most_sixes": most_sixes.to_dict(),
        "best_economy": best_economy[['economy']].to_dict(orient="index"),
        "best_finisers": best_finisers.to_dict(orient="index")
    }

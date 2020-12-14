import pandas as pd
import numpy as np
import multiprocessing
import time
import os


def normalizeData(week):
    prev_time = time.time()
    print(f"Week{week} | Started script | {round(time.time()-prev_time, 4)} s")
    prev_time = time.time()

    track_df = pd.read_csv(f"./data/week{week}.csv", low_memory=False).reset_index(drop=True)
    games_df = pd.read_csv("./data/games.csv", low_memory=False).reset_index(drop=True)
    plays_df = pd.read_csv("./data/plays.csv", low_memory=False).reset_index(drop=True)

    print(f"Week{week} | Done reading {len(track_df)} rows | {round(time.time()-prev_time, 4)} s")
    prev_time = time.time()

    track_df['nflId'] = track_df.nflId.fillna(0)  # add an ID to the football rows

    track_df = track_df.join(games_df[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']].set_index('gameId'), on=['gameId'])
    track_df['teamAbbr'] = np.select(
        [track_df.team == 'home', track_df.team == 'away'],
        [track_df.homeTeamAbbr, track_df.visitorTeamAbbr],
        default='FTBL')

    track_df = track_df.join(plays_df[['gameId', 'playId', 'possessionTeam']
                                      ].set_index(['gameId', 'playId']), on=['gameId', 'playId'])
    track_df['team_pos'] = np.select([track_df.teamAbbr == 'FTBL', track_df.teamAbbr ==
                                      track_df.possessionTeam, True], ['FTBL', 'OFF', 'DEF'])

    track_df = track_df.drop(columns=['homeTeamAbbr', 'visitorTeamAbbr', 'possessionTeam'])

    los = track_df.loc[(track_df.displayName == 'Football') & (track_df.event == 'ball_snap')][[
        'gameId', 'playId', 'x']].rename({'x': 'los'}, axis=1).set_index(['gameId', 'playId'])
    track_df = track_df.join(los, on=['gameId', 'playId'])
    track_df.reset_index(drop=True, inplace=True)

    print(f"Week{week} | Done joins | {round(time.time()-prev_time, 4)} s")
    prev_time = time.time()

    track_df['los_norm'] = track_df.los
    track_df.loc[track_df.playDirection == 'left', 'los_norm'] = 120-track_df.los_norm

    track_df['x_norm'] = track_df['x']-(track_df['los']-track_df['los_norm'])
    track_df['y_norm'] = track_df['y']
    track_df['o_norm'] = track_df['o']
    track_df['dir_norm'] = track_df['dir']
    track_df.reset_index(drop=True, inplace=True)
    track_df.loc[track_df.playDirection == 'left', 'x_norm'] = (
        track_df['los_norm']-track_df['x_norm'])+track_df['los_norm']
    track_df.loc[track_df.playDirection == 'left', 'y_norm'] = (53.3/2-track_df['y_norm'])+53.3/2
    track_df.loc[track_df.playDirection == 'left', 'o_norm'] = (track_df['o_norm']+180) % 360
    track_df.loc[track_df.playDirection == 'left', 'dir_norm'] = (track_df['dir_norm']+180) % 360
    track_df = track_df.drop(['x', 'y', 'o', 'dir', 'los', 'playDirection'], axis=1).rename(
        columns={'x_norm': 'x', 'y_norm': 'y', 'o_norm': 'o', 'dir_norm': 'dir', 'los_norm': 'los'})

    deltaT = 0.1
    track_df = track_df.rename(columns={'dir': 's_dir'})

    track_df['s_dir_rad'] = track_df.s_dir*np.pi/180

    track_df['v_x'] = np.sin(track_df.s_dir_rad) * track_df.s
    track_df['v_y'] = np.cos(track_df.s_dir_rad) * track_df.s

    track_df.loc[track_df.nflId == 0, 'v_x'] = track_df.loc[track_df.nflId == 0].groupby(['gameId', 'playId'])[
        'x'].diff().fillna(0)/deltaT
    track_df.loc[track_df.nflId == 0, 'v_y'] = track_df.loc[track_df.nflId == 0].groupby(['gameId', 'playId'])[
        'y'].diff().fillna(0)/deltaT

    track_df['v_mag'] = np.sqrt(track_df.v_x*track_df.v_x+track_df.v_y*track_df.v_y)
    track_df['v_theta'] = np.arctan(track_df.v_y/track_df.v_x).fillna(0)

    track_df['a_x'] = track_df.groupby(
        ['gameId', 'playId', 'nflId'])['v_x'].transform(
        lambda x: x.diff().fillna(0)) / deltaT
    track_df['a_y'] = track_df.groupby(
        ['gameId', 'playId', 'nflId'])['v_y'].transform(
        lambda x: x.diff().fillna(0)) / deltaT
    track_df['a_theta'] = np.arctan(track_df.a_y/track_df.a_x).fillna(0)
    track_df['a_mag'] = np.sqrt(track_df.a_x*track_df.a_x+track_df.a_y*track_df.a_y)

    track_df = track_df.rename(columns={'a': 'a_old'})
    track_df = track_df.round(2)

    mapping = {'DB': 'DB', 'CB': 'DB', 'S': 'S', 'SS': 'S', 'FS': 'S', 'WR': 'WR', 'MLB': 'LB', 'OLB': 'LB',
               'ILB': 'LB', 'LB': 'LB', 'DL': 'DL', 'DT': 'DL', 'DE': 'DL', 'NT': 'DL', 'QB': 'QB', 'RB': 'RB',
               'HB': 'RB', 'TE': 'TE', 'P': 'ST', 'K': 'ST', 'LS': 'ST', 'FB': 'FB'}
    track_df['position_general'] = track_df.position.map(mapping)

    track_df = track_df[['gameId', 'playId', 'frameId', 'event', 'nflId', 'displayName', 'jerseyNumber', 'position',
                         'position_general', 'team', 'team_pos', 'teamAbbr', 'route', 'time', 'los', 'x', 'y', 'dis',
                         'o', 's', 's_dir', 's_dir_rad', 'v_x', 'v_y', 'v_theta', 'v_mag', 'a_old', 'a_x', 'a_y',
                         'a_theta', 'a_mag']]

    print(f"Week{week} | Done normalizing | {round(time.time()-prev_time,4)} s")
    prev_time = time.time()

    f = f'./data/week{week}_norm.csv'
    track_df.to_csv(f, index=False)

    print(f"Week{week} | Done exporting | Took {round(time.time()-prev_time, 4)} s")
    prev_time = time.time()


WEEKS = list(range(1, 18))
# WEEKS = ['toy']
pool = multiprocessing.Pool(10)
pool.map(normalizeData, WEEKS)
pool.close()

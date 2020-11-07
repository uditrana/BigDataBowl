import pandas as pd
import multiprocessing
import time
import os
# pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


def normalizeData(week):
    prev_time = time.time()
    print(f"Week{week} | Started script | {round(time.time()-prev_time, 4)} s")
    prev_time = time.time()
    data = pd.read_csv(f"./data/week{week}.csv", low_memory=False)
    print(f"Week{week} | Done reading {len(data)} rows | {round(time.time()-prev_time, 4)} s")
    prev_time = time.time()

    data.reset_index(drop=True, inplace=True)

    los = data.loc[(data.displayName == 'Football') & (data.event == 'ball_snap')][[
        'gameId', 'playId', 'x']].rename({'x': 'los'}, axis=1).set_index(['gameId', 'playId'])
    data = data.join(los, on=['gameId', 'playId'])
    data.reset_index(drop=True, inplace=True)

    print(f"Week{week} | Done LOS-join week | {round(time.time()-prev_time, 4)} s")
    prev_time = time.time()

    data['los_norm'] = 40
    data['x_norm'] = data['x']-(data['los']-data['los_norm'])
    data['y_norm'] = data['y']
    data['o_norm'] = data['o']
    data['dir_norm'] = data['dir']
    data.reset_index(drop=True, inplace=True)
    data.loc[data.playDirection == 'left', 'x_norm'] = (data['los_norm']-data['x_norm'])+data['los_norm']
    data.loc[data.playDirection == 'left', 'y_norm'] = (53.3/2-data['y_norm'])+53.3/2
    data.loc[data.playDirection == 'left', 'o_norm'] = (data['o_norm']+180) % 360
    data.loc[data.playDirection == 'left', 'dir_norm'] = (data['dir_norm']+180) % 360

    print(f"Week{week} | Done normalizing | {round(time.time()-prev_time,4)} s")
    prev_time = time.time()

    f = f'./data/week{week}_norm.csv'
    data.to_csv(f, index=False)

    print(f"Week{week} | Done exporting | Took {round(time.time()-prev_time, 4)} s")
    prev_time = time.time()


WEEKS = list(range(1, 18))
# WEEKS = ['toy']
pool = multiprocessing.Pool(8)
pool.map(normalizeData, WEEKS)
pool.close()

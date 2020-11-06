import pandas as pd
import multiprocessing
import time
import os
# pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)


def normalizeData(week):
    prev_time = time.time()
    print(
        f"Week{week} | Started script | {round(time.time()-prev_time, 4)} s")
    prev_time = time.time()
    data = pd.read_csv(f"./data/week{week}.csv", low_memory=False)
    print(
        f"Week{week} | Done reading {len(data)} rows | {round(time.time()-prev_time, 4)} s")
    prev_time = time.time()

    data.reset_index(drop=True, inplace=True)

    los = data.loc[(data.displayName == 'Football') & (data.event == 'ball_snap')][[
        'gameId', 'playId', 'x']].rename({'x': 'los'}, axis=1).set_index(['gameId', 'playId'])
    data = data.join(los, on=['gameId', 'playId'])
    data.reset_index(drop=True, inplace=True)

    print(
        f"Week{week} | Done LOS-join week | {round(time.time()-prev_time, 4)} s")
    prev_time = time.time()

    # def normRows(r):
    #     # translate los
    #     r['los_norm'] = 40
    #     r['x_norm'] = r['x']-(r['los']-r['los_norm'])
    #     r['y_norm'] = r['y']

    #     # flip field if play is left
    #     if r['playDirection'] == 'left':
    #         r['x_norm'] = (r['los_norm']-r['x_norm'])+r['los_norm']
    #         r['y_norm'] = (53.3/2-r['y_norm'])+53.3/2

    #     return r
    # data = data.apply(lambda r: normRows(r), axis=1)

    data['los_norm'] = 40
    data['x_norm'] = data['x']-(data['los']-data['los_norm'])
    data['y_norm'] = data['y']
    data.reset_index(drop=True, inplace=True)
    data.loc[data.playDirection == 'left', 'x_norm'] = (
        data['los_norm']-data['x_norm'])+data['los_norm']
    data.loc[data.playDirection == 'left', 'y_norm'] = (
        53.3/2-data['y_norm'])+53.3/2

    print(
        f"Week{week} | Done normalizing | {round(time.time()-prev_time,4)} s")
    prev_time = time.time()

    f = f'./data_norm2/week{week}_norm.csv'
    data.to_csv(f, index=False)

    print(
        f"Week{week} | Done exporting | Took {round(time.time()-prev_time, 4)} s")
    prev_time = time.time()


WEEKS = list(range(1, 18))
# WEEKS = ['toy']
pool = multiprocessing.Pool(8)
pool.map(normalizeData, WEEKS)
pool.close()

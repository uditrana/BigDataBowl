import pandas as pd
from models.play_eppa_gpu import *
from pathlib import Path
import traceback
import os
from tqdm import tqdm
from utils import get_repo_dir


WEEK_START = 1
WEEK_END = 2

out_dir_path = os.path.join(get_repo_dir(), 'output/{}')  # for cloud runs
path_shared = os.path.join(get_repo_dir(), 'data/{}')

# main loop
for week in range(WEEK_START, WEEK_END):
    track_df = pd.read_csv(
        path_shared.format(
            f'week{week}_norm.csv',
            usecols=['nflId', 'displayName', 'position', 'team_pos', 'x', 'y', 'v_x', 'v_y', 'v_mag', 'v_theta', 'a_x', 'a_y', 'a_mag', 'a_theta']))

    plays = sorted(list(set(map(lambda x: (x[0].item(), x[1].item()), track_df.groupby(
        ['gameId', 'playId'], as_index=False).first()[['gameId', 'playId']].to_numpy()))))

    fails = []
    Path(out_dir_path.format(f'{week}')).mkdir(parents=True, exist_ok=True)
    with open(out_dir_path.format(f'{week}/errors.txt'), 'w+') as f:
        # for (gid, pid) in tqdm(random.sample(plays, len(plays))):
        for (gid, pid) in tqdm(plays):
            dir = out_dir_path.format(f'{week}/{gid}/{pid}')
            if os.path.exists(dir):
                print(f'EXISTS: {gid}, {pid}')
            else:
                try:
                	play_eppa_gpu(track_df, gid, pid, viz_df=False, save_np=False, stats_df=True,
                                  viz_true_proj=True, save_all_dfs=True, out_dir_path=out_dir_path)
                except Exception as e:
                    fails.append((gid, pid))
                    f.write(f"\nERROR: {gid}, {pid}\n")
                    f.write(traceback.format_exc()+'\n\n\n')
                    print(f"ERROR: {gid}, {pid}. {e}")
        print(len(plays))
        print(len(fails))
        f.write(f"{len(fails)} out of {len(plays)}")
        f.write(str(fails))

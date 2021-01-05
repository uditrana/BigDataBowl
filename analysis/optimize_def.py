# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import torch
from models.params import params
from models.consts import *
from models.frame_eppa import frame_eppa, set_device
from tqdm import tqdm


out_dir_path = '../def_opt_output/{}'  # for cloud runs
path_shared = '../data/{}'

plan_horizon = 0.8  # how far ahead to project (in seconds) while optimizing path
plan_res = 0.1  # how long (in seconds) to execute plan. less than plan_horizon
frame_dt = 0.1  # how far apart (in seconds) the frames are
plan_res_linspace = np.linspace(frame_dt, plan_res, np.round(plan_res/frame_dt).astype(int))[:,None]
# 10*dt is the number of frames to increment by
res_incr = int(10 * plan_res)
horizon_incr = int(10 * plan_horizon)
reduce_eppa_mode = 'sum'  # choose from {'sum', 'max', 'softmax'}
softmax_temp = 1000  # used for 'softmax' reduce_eppa_mode


week_game_plays = [
    [1, 2018090600, 75],        # jenkins push julio ob 10 yd
    [13, 2018120212, 1834],     # gronk seam
    [15, 2018121609, 1577],     # doug 9ers
    [6, 2018101404, 977],       # 4 verts
    [4, 2018093004, 5602],      # 4 hitches
    [13, 2018120212, 2971]      # brady flash seam zone
]
# WEEK, game_id, play_id = week_game_plays[0]


def reduce_eppa(frame_eppa, mode=reduce_eppa_mode):
    """ takes in (F, T) frame_eppa and reduces to scalar """
    if mode == 'sum':
        return frame_eppa.sum()
    elif mode == 'max':
        return frame_eppa.max()
    elif mode == 'softmax':
        frame_eppa_torch = torch.from_numpy(frame_eppa.flatten())
        return (torch.softmax(frame_eppa_torch * softmax_temp, dim=0) \
                * frame_eppa_torch).sum()


def optimize_def_frame(play_df, frame_id, frame_only):
    """
    :param play_df: pandas df containing all tracking info for this play
    :param frame_id: weirdly this is the frame the defensive player is projecting TO.
        the eppa will be evaluated at frame (frame_id - horizon_incr + res_incr).
    :param frame_only: return eppa for this frame without optimizing for proj values
    this function:
        1. projects offensive players forward a frame
        2. loops over defensive players from LoS to defensive backfield, and
            * finds the viable field locations the player can end at
            * checks eppa for each possible position
            * greedily sets the player's location to minimize conceded EPPA
        3. returns a tuple consisting of
            * updated_def_pos: an object like def_pos with updated positions
            * eppa: a float representing EPPA conceded on that frame
    Assumptions:
        * Each player can sustain their instantaneous acceleration for plan_horizon time
    """
    if frame_only:
        return np.array([reduce_eppa(frame_eppa(play_df, frame_id))])
    # frame ids to optimize
    frame_ids_opt = list(range(frame_id-horizon_incr+1, frame_id-horizon_incr+res_incr+1))
    ### 1. assign proj pos/vel/acc from prev frame. these are the def players'
    # "thoughts" as they "decide" (are optimized on) where to go. projections
    # are computed for all players, but are only placeholders for defensive
    # players since they'll eventually be updated sequentially in this func
    prev_df = play_df.loc[play_df.frameId == frame_id - horizon_incr]
    # project location forward
    proj_x_loc = prev_df.x_opt + prev_df.v_x_opt*plan_horizon + 0.5*prev_df.a_x_opt*plan_horizon**2
    proj_y_loc = prev_df.y_opt + prev_df.v_y_opt*plan_horizon + 0.5*prev_df.a_y_opt*plan_horizon**2
    # project offensive velocity forward
    proj_x_vel = prev_df.v_x_opt + prev_df.a_x_opt*plan_horizon
    proj_y_vel = prev_df.v_y_opt + prev_df.a_y_opt*plan_horizon
    # clip velocity so that speed is clipped at s_max
    proj_vel_mag = np.linalg.norm((proj_x_vel, proj_y_vel), axis=0)  # (J,)
    adj_factor = params.s_max / proj_vel_mag[proj_vel_mag > params.s_max]
    proj_x_vel[proj_vel_mag > params.s_max] *= adj_factor
    proj_y_vel[proj_vel_mag > params.s_max] *= adj_factor
    # assign clipped proj vel back in
    play_df.loc[play_df.frameId == frame_id,
        ['x_opt', 'y_opt', 'v_x_opt', 'v_y_opt', 'a_x_opt', 'a_y_opt']] = \
        np.stack((proj_x_loc.values,
                    proj_y_loc.values,
                    proj_x_vel.values,
                    proj_y_vel.values,
                    prev_df.a_x_opt.values,
                    prev_df.a_y_opt.values)).T
    ### 2. optimize defensive players in order from LoS to defensive backfield
    # use their location at prev frame to start and decide where they should want to end up now
    def_df = play_df.loc[(play_df.frameId == frame_id - horizon_incr) & (play_df.team_pos == 'DEF')]\
                .sort_values('x_opt', ascending=True)
    for def_row in def_df.itertuples():
        def_id = def_row.nflId
        # def loc is where the opt path put him at the prev frame
        def_loc = np.array([def_row.x_opt, def_row.y_opt], dtype=np.float32)
        # def vel is the velocity he had at the prev frame under the opt path
        def_vel = np.array([def_row.v_x_opt, def_row.v_y_opt], dtype=np.float32)
        def_vel = np.array([5, 6], dtype=np.float32)
        reach_vecs = field_locs - def_loc  # (F, 2)
        reach_accs = 2 * (reach_vecs - def_vel * plan_horizon) / plan_horizon**2  # (F, 2); required accel to get to each spot
        reach_vels = reach_accs * plan_horizon + def_vel  # (F, 2); resultant velocity at each spot
        reach_acc_mags = np.linalg.norm(reach_accs, axis=1)  # (F,)
        reach_vel_mags = np.linalg.norm(reach_vels, axis=1)  # (F,)
        reachable_idx = (reach_acc_mags < params.a_max) & (reach_vel_mags < params.s_max)
        reachable_accs = reach_accs[reachable_idx]
        reachable_vels = reach_vels[reachable_idx]  # (R, 2); R is number of reachable spots
        reachable_locs = field_locs[reachable_idx]  # (R, 2); R is number of reachable spots
        frame_eppa_vals = []
        # print(f'{len(reachable_locs)} reachable locations found')
        for i in range(len(reachable_locs)):
            play_df.loc[(play_df.frameId == frame_id) & (play_df.nflId == def_id),
                        ['x_opt', 'y_opt', 'v_x_opt', 'v_y_opt', 'a_x_opt', 'a_y_opt']] = \
                        reachable_locs[i, 0], reachable_locs[i, 1], reachable_vels[i, 0],\
                        reachable_vels[i, 1], reachable_accs[i, 0], reachable_accs[i, 1]
            frame_eppa_val = reduce_eppa(frame_eppa(play_df, frame_id))
            frame_eppa_vals.append(frame_eppa_val)
        opt_idx = np.array(frame_eppa_vals).argmin()
        # get the constant acceleration vector for this optimal path
        final_def_acc = reachable_accs[opt_idx]
        # we optimized path for PLAN_HORIZON-second lookahead but are only executing for PLAN_RES seconds
        final_def_loc = def_loc[None] + def_vel[None] * plan_res_linspace + 0.5 * final_def_acc[None] * plan_res_linspace**2
        final_def_vel = def_vel[None] + final_def_acc[None] * plan_res_linspace
        final_def_acc = np.broadcast_to(final_def_acc, final_def_vel.shape)
        # NOTE below assumes frameId is sorted increasing
        play_df.loc[(play_df.frameId.isin(frame_ids_opt)) & (play_df.nflId == def_id), ['x_opt', 'y_opt', 'v_x_opt', 'v_y_opt', 'a_x_opt', 'a_y_opt']]\
                = np.concatenate((final_def_loc, final_def_vel, final_def_acc), axis=1)
    ### 3. return final optimized eppa for this frame
    return np.array([reduce_eppa(frame_eppa(play_df, frame_id_opt)) for frame_id_opt in frame_ids_opt])


def optimize_def(week, game_id, play_id):
    track_df = pd.read_csv(path_shared.format(f'week{week}_norm.csv',
        usecols=['nflId', 'displayName', 'position', 'team_pos', 'x', 'y', 'v_x', 'v_y', 'v_mag', 'v_theta', 'a_x', 'a_y', 'a_mag', 'a_theta']))
    play_df = track_df.loc[(track_df.gameId == game_id) & (track_df.playId == play_id)]
    # initialize the optimal values
    play_df['x_opt'] = play_df.x
    play_df['y_opt'] = play_df.y
    play_df['v_x_opt'] = play_df.v_x
    play_df['v_y_opt'] = play_df.v_y
    play_df['a_x_opt'] = play_df.a_x
    play_df['a_y_opt'] = play_df.a_y
    ball_snap_frame = play_df.loc[(play_df.nflId == 0) & (play_df.event == 'ball_snap')].frameId.iloc[0].astype(int)
    pass_forward_frame = play_df.loc[(play_df.nflId == 0) & ((play_df.event == 'pass_forward') |
                                                             (play_df.event == 'pass_shovel'))].frameId.sort_values().iloc[0].astype(int)
    # necessary for frame_eppa function
    play_df['frames_since_snap'] = play_df.frameId - ball_snap_frame
    # TODO force add in pass_forward_frame
    frame_ids = list(np.arange(ball_snap_frame+horizon_incr, pass_forward_frame+horizon_incr-res_incr+1, res_incr))
    eppas = np.concatenate([optimize_def_frame(play_df, ball_snap_frame, frame_only=True)]\
                + [optimize_def_frame(play_df, int(frame_id), frame_only=False)
                    for frame_id in tqdm(frame_ids)])
    return play_df, eppas

def run(week, game_id, play_id, save=True, device='cuda'):
    set_device(device)
    optimized_play_df, eppas = optimize_def(week, game_id, play_id)
    if save:
        play_dir = out_dir_path.format(f'def_opt_{game_id}_{play_id}')
        os.makedirs(play_dir, exist_ok=True)
        optimized_play_df.to_csv(os.path.join(play_dir, 'opt_play_df.csv'))
        np.save(os.path.join(play_dir, 'eppas.npy'), eppas)
    print(f'Finished running week {week}, game {game_id}, play {play_id}!')
    return optimized_play_df, eppas

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--week', type=int, nargs='?', help='week of game')
    parser.add_argument('-g', '--game', type=int, nargs='?', help='game ID')
    parser.add_argument('-p', '--play', type=int, nargs='?', help='play ID')
    parser.add_argument('-n', '--nosave', action='store_true', help='won\'t save if set')
    args = parser.parse_args()

    week, game, play, save = args.week, args.game, args.play, not args.nosave
    if week is None or game is None or play is None:
        if not (response := input('One or more of week, game, play was unspecified. Run all? [Y/n]: ').lower()) or response[0] != 'n':
            import multiprocessing as mp; mp.set_start_method('spawn')
            with mp.Pool(processes=min(16, len(week_game_plays))) as pool:
                pool.starmap(run, [week_game_play + [save, f'cuda:{idx % torch.cuda.device_count()}'] for idx, week_game_play in enumerate(week_game_plays)])        
        else:
            exit()
    else:
        run(week, game, play, save)


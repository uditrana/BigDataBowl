
import traceback
import os
import treelite
import random
import treelite_runtime
import joblib
import xgboost as xgb
from tqdm import tqdm

import numpy as np
import pandas as pd
from pathlib import Path
import torch


WEEK = 1

# file loading and prep
path_shared = '../data/{}'

games_df = pd.read_csv(path_shared.format('games.csv'))
plays_df = pd.read_csv(path_shared.format('plays.csv'))
players_df = pd.read_csv(path_shared.format('players.csv'))
track_df = pd.read_csv(
    path_shared.format(
        f'week{WEEK}_norm.csv',
        usecols=['nflId', 'displayName', 'position', 'team_pos', 'x', 'y', 'v_x', 'v_y', 'v_mag', 'v_theta', 'a_x',
                 'a_y', 'a_mag', 'a_theta']))
pbp_2018 = pd.read_csv(path_shared.format('play_by_play_2018.csv'), low_memory=False)
pbp_joined = pd.merge(plays_df, pbp_2018, how="left", left_on=["gameId", "playId"], right_on=["old_game_id", "play_id"])

pbp_joined["retractable"] = np.where(((pbp_joined['roof'] == "open ") | (pbp_joined['roof'] == "closed")), 1, 0)
pbp_joined["dome"] = np.where((pbp_joined['roof'] == "dome"), 1, 0)
pbp_joined["outdoors"] = np.where((pbp_joined['roof'] == "outdoors"), 1, 0)
pbp_joined["era0"] = 0
pbp_joined["era1"] = 0
pbp_joined["era2"] = 0
pbp_joined["era3"] = 0
pbp_joined["era4"] = 1
pbp_joined["down1"] = np.where((pbp_joined['down_x'] == 1), 1, 0)
pbp_joined["down2"] = np.where((pbp_joined['down_x'] == 2), 1, 0)
pbp_joined["down3"] = np.where((pbp_joined['down_x'] == 3), 1, 0)
pbp_joined["down4"] = np.where((pbp_joined['down_x'] == 4), 1, 0)
pbp_joined["home"] = np.where((pbp_joined['posteam'] == pbp_joined['home_team']), 1, 0)

out_dir_path = '../output/{}'  # for cloud runs

# rerun cell if xgboost loading isnt working for your machine (needs xgboost 1.2.1 exactly)
#bst = joblib.load("./in/xyac_model.model")
#xgb.plot_importance(bst)
#scores = bst.get_score(importance_type='gain')
#print(scores.keys())
#cols_when_model_builds = bst.feature_names
#model = treelite.Model.from_xgboost(bst)
#toolchain = 'gcc'
#model.export_lib(toolchain=toolchain, libpath='./in/xyacmymodel.so',
#                 params={'parallel_comp': 32}, verbose=True)  # .so for ubuntu, .dylib for mac
#
#bst = joblib.load("./in/epa_model_rishav_no_time.model")
#xgb.plot_importance(bst)
#scores = bst.get_score(importance_type='gain')
#print(scores.keys())
#cols_when_model_builds = bst.feature_names
#model = treelite.Model.from_xgboost(bst)
#toolchain = 'gcc'
#model.export_lib(toolchain=toolchain, libpath='./in/epa_no_time_mymodel.so',
#                 params={'parallel_comp': 32}, verbose=True)  # .so for ubuntu, .dylib for mac


def params(): return None  # create an empty object to add params


params.a_max = 7.68
params.s_max = 8.98
# params.reax_t = params.s_max/params.a_max
params.reax_t = 0.3
params.tti_sigma = 0.3
# params.tti_sigma = 0.45
params.cell_length = 1
params.alpha = 1.0
params.z_min = 1
params.z_max = 3
params.epsilon = 0.0
params.catch_lambda = 0.6
vars(params)

dt = np.float64
dt_torch = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def torchify(np_arr):
    return torch.from_numpy(np_arr).to(device)

# model constants
T = np.linspace(0.1, 4, 40, dtype=dt)
x = np.linspace(0.5, 119.5, 120, dtype=dt)
y = np.linspace(-0.5, 53.5, 55, dtype=dt)
y[0] = -0.2
xx, yy = np.meshgrid(x, y)
field_locs = np.stack((xx, yy)).reshape(2, -1).T  # (F, 2)
tot_pass_cnt = len(field_locs[:, 1])*len(T)
print(f'Considering {tot_pass_cnt} passes per frame')
T_torch = torchify(T)
xx_torch = torchify(xx)
yy_torch = torchify(yy)
field_locs_torch = torchify(field_locs)


# historical trans model inputs/params
L_given_ts = np.load('in/L_given_t.npy')
T_given_Ls = pd.read_csv('in/T_given_L.csv')['p'].values.reshape(60, len(T))  # (61, T)
# from L_given_t in historical notebook
x_min, x_max = -9, 70
y_min, y_max = -39, 40
t_min, t_max = 10, 63

# epa/xyac model loading
bst = joblib.load("./in/xyac_model.model")
scores = bst.get_score(importance_type='gain')
cols_when_model_builds = bst.feature_names
xyac_predictor = treelite_runtime.Predictor('./in/xyacmymodel.so')
epa_model = joblib.load("./in/epa_model_rishav_no_time.model")
scores = epa_model.get_score(importance_type='gain')
cols_when_model_builds_ep = epa_model.feature_names
epa_predictor = treelite_runtime.Predictor('./in/epa_no_time_mymodel.so')


def play_eppa(game_id, play_id, viz_df=False, save_np=False, stats_df=False, viz_true_proj=False, save_all_dfs=False):
    play_df = track_df[(track_df.playId == play_id) & (track_df.gameId == game_id)].sort_values(by='frameId')

    ball_snap_frame = play_df.loc[(play_df.nflId == 0) & (play_df.event == 'ball_snap')].frameId.iloc[0]
    pass_forward_frame = play_df.loc[(play_df.nflId == 0) & ((play_df.event == 'pass_forward') | (
        play_df.event == 'pass_shovel') | (play_df.event == 'qb_sack'))].frameId.iloc[0]
    play_df['frames_since_snap'] = play_df.frameId - ball_snap_frame

    pass_arriveds = play_df.loc[(play_df.nflId == 0) & (play_df.event == 'pass_arrived'), 'frameId'].to_list()
    true_pass_found = len(pass_arriveds) > 0

    if true_pass_found:
        pass_arrived_frame = pass_arriveds[0]
        true_T_frames = pass_arrived_frame - pass_forward_frame
        true_x, true_y = play_df.loc[(play_df.nflId == 0) & (play_df.event == 'pass_arrived'),
                                     ['x', 'y']].iloc[0].to_numpy(dtype=dt)
        print(f"True Pass: t: {pass_forward_frame} x:{true_x} y:{true_y} T:{true_T_frames/10}")
        true_T_idx = np.rint(true_T_frames).astype(int)-1
        true_x_idx = (true_x).astype(int)
        true_y_idx = (true_y).astype(int)+1
        true_f_idx = np.ravel_multi_index((true_y_idx, true_x_idx), xx.shape)
        (true_x_idxd, true_y_idxd), true_T_idxd = field_locs[true_f_idx], T[true_T_idx]
        print(f"True Pass idxs: t: {pass_forward_frame} x:{true_x_idx} y:{true_y_idx} f:{true_f_idx} T:{true_T_idx}")
        print(f"True Pass idxd: t: {pass_forward_frame} x:{true_x_idxd} y:{true_y_idxd} T:{true_T_idxd}")
    else:
        print(f"No True Pass Found: No pass_arrived event \n")

    # per play epa model
    def getEPAModel():
        epvals = np.array([7, -7, 3, -3, 2, -2, 0])
        joined_df = pbp_joined[(pbp_joined.playId == play_id) & (pbp_joined.gameId == game_id)]
        epa_df = pd.DataFrame({'play_endpoint_x': x})

        test = {}
        for feat in epa_model.feature_names:
            test[feat] = [joined_df.iloc[0][feat]]
            epa_df[feat] = joined_df.iloc[0][feat]

        first_df = pd.DataFrame(test)

        dtest = treelite_runtime.Batch.from_npy2d(first_df[cols_when_model_builds_ep].values)
        ypred = epa_predictor.predict(dtest)
        ep = np.sum(ypred*epvals, axis=1)

        epa_df["before_play_ep"] = ep[0]

        epa_df['los'] = 110 - epa_df['yardline_100']
        epa_df['first_down_line'] = epa_df['los'] + epa_df["ydstogo"]
        epa_df['expected_end_up_line'] = epa_df['play_endpoint_x']

        epa_df['isFirstDown'] = 0
        epa_df['isFirstDown'] = np.where(epa_df['expected_end_up_line'] >= epa_df['first_down_line'], 1, 0)

        epa_df['yardline_100'] = np.round(110 - epa_df['expected_end_up_line'])
        epa_df['yardline_100'] = np.clip(epa_df['yardline_100'], 0, 100)
        epa_df['ydstogo'] = epa_df['first_down_line'] - epa_df['expected_end_up_line']
        epa_df['ydstogo'] = np.where(epa_df['isFirstDown'] == 1, 10, epa_df['ydstogo'])
        old_down = joined_df.iloc[0]['down_x']

        for d in range(1, 6):
            epa_df['down%d' % d] = 1 if (d == old_down+1) else 0

        epa_df['down1'] = np.where(epa_df['isFirstDown'] == 1, 1, epa_df['down1'])
        epa_df['down2'] = np.where(epa_df['isFirstDown'] == 1, 0, epa_df['down2'])
        epa_df['down3'] = np.where(epa_df['isFirstDown'] == 1, 0, epa_df['down3'])
        epa_df['down4'] = np.where(epa_df['isFirstDown'] == 1, 0, epa_df['down4'])
        epa_df['down5'] = np.where(epa_df['isFirstDown'] == 1, 0, epa_df['down5'])  # down 5 means turnover on downs

        # offense is other team inverted if turnover on downs
        epa_df['yardline_100'] = np.where(epa_df.down5 == 1, 100-epa_df['yardline_100'], epa_df['yardline_100'])
        epa_df['ydstogo'] = np.where(epa_df.down5 == 1, 10, epa_df['ydstogo'])
        epa_df['down1'] = np.where(epa_df.down5 == 1, 1, epa_df['down1'])

        ### UPDATE EPA VARIABLES ###

        dtest = treelite_runtime.Batch.from_npy2d(epa_df[cols_when_model_builds_ep].values)
        ypred = epa_predictor.predict(dtest)
        ep = np.sum(ypred*epvals, axis=1)
        epa_df['xep'] = ep  # ep after play
        # SCORE SAFETIES
        epa_df['xep'] = np.where(epa_df['play_endpoint_x'] <= 10, -2, epa_df['xep'])
        # SCORE Tds
        epa_df['xep'] = np.where(epa_df['play_endpoint_x'] >= 110, 7, epa_df['xep'])

        epa_df['xepa'] = np.where(epa_df.down5 == 1, -epa_df['xep'] - epa_df['before_play_ep'],
                                  epa_df['xep'] - epa_df['before_play_ep'])  # if turnover

        only_vals = epa_df[["play_endpoint_x", "xep", "xepa"]]  # THIS CONTAINS THE EPA VALUES BASED ON PLAY ENDPOINT
        return only_vals

    epa_df = getEPAModel()

    # input: tracking data for a single frame
    # output: frame_eppa (F, T)... (writes intermediate F,T to disk)

    def frame_eppa(frame_id):
        nonlocal play_df

        frame_df = play_df.loc[play_df.frameId == frame_id].reset_index()
        ball_start = frame_df.loc[frame_df.position == 'QB', ['x', 'y']].iloc[0].to_numpy(dtype=dt)
        ball_start_torch = torchify(ball_start)
        t = frame_df.frames_since_snap.iloc[0]
        frame_df = frame_df.loc[(frame_df.nflId != 0) & (frame_df.position != 'QB')]  # remove ball and qb from df

        reach_vecs = (field_locs - ball_start).astype(dt)  # (F, 2)
        reach_vecs_torch = torchify(reach_vecs)

        frame_df = frame_df.drop_duplicates(subset='nflId').sort_values('nflId').reset_index()

        # project motion by reaction time
        frame_df['v_x_r'] = frame_df.a_x*params.reax_t+frame_df.v_x
        frame_df['v_y_r'] = frame_df.a_y*params.reax_t+frame_df.v_y
        frame_df['v_r_mag'] = np.linalg.norm(np.array([frame_df.v_x_r, frame_df.v_y_r], dtype=dt), axis=0)
        frame_df['v_r_theta'] = np.arctan2(frame_df.v_y_r, frame_df.v_x_r)
        frame_df['x_r'] = frame_df.x + frame_df.v_x*params.reax_t + 0.5*frame_df.a_x*params.reax_t**2
        frame_df['y_r'] = frame_df.y + frame_df.v_y*params.reax_t + 0.5*frame_df.a_y*params.reax_t**2

        player_teams = frame_df['team_pos'].to_numpy()  # J,
        player_off = player_teams == 'OFF'
        player_def = np.logical_not(player_off)
        player_team_names = frame_df['teamAbbr'].to_numpy()  # J,
        player_ids = frame_df['nflId'].to_numpy()
        player_names = frame_df['displayName'].to_numpy()
        reaction_player_locs = frame_df[['x_r', 'y_r']].to_numpy(dtype=dt)  # (J, 2)
        reaction_player_vels = frame_df[['v_x_r', 'v_y_r']].to_numpy(dtype=dt)  # (J,2)

        player_off_torch = torchify(player_off)  # J,
        player_def_torch = torchify(player_def)  # J,

        # intercept vector between each player and field location
        int_d_vec = field_locs[:, None, :] - reaction_player_locs  # F, J, 2
        int_d_mag = np.linalg.norm(int_d_vec, axis=2)  # F, J
        int_d_theta = np.arctan2(int_d_vec[..., 1], int_d_vec[..., 0])
        # projecting player velocity on d_vec to get initial speed along d_vec
        int_s0 = np.clip(np.sum(int_d_vec*reaction_player_vels, axis=2)/int_d_mag, -params.s_max, params.s_max)  # F, J,

        # calculate time to int to ball loc on d_vec based on phys model
        t_lt_smax = (params.s_max-int_s0)/params.a_max  # F, J,
        d_lt_smax = t_lt_smax*((int_s0+params.s_max)/2)  # F, J,
        # if accelerating would overshoot, then t = -v0/a + sqrt(v0^2/a^2 + 2x/a) (from kinematics)
        t_lt_smax = np.where(d_lt_smax > int_d_mag, -int_s0/params.a_max+np.sqrt((int_s0/params.a_max)
                                                                                 ** 2+2*int_d_mag/params.a_max), t_lt_smax)
        d_lt_smax = np.clip(d_lt_smax, 0, int_d_mag)
        d_at_smax = int_d_mag - d_lt_smax  # F, J,
        t_at_smax = d_at_smax/params.s_max  # F, J,
        t_tot = t_lt_smax+t_at_smax+params.reax_t  # F, J,

        # int success if T-t_tot = dT <  0. Put through sigmoid to add temporal uncertainty around
        int_dT = T[None, :, None] - t_tot[:, None, :] + params.epsilon  # F, T, J
        p_int = (1/(1. + np.exp(-np.pi/np.sqrt(3.0)/params.tti_sigma * int_dT, dtype=dt)))  # F, T, J
        p_int_adj = p_int.copy()
        p_int_def = 1-np.prod((1-p_int_adj[:, :, player_def]), axis=-1)  # F, T,
        p_int_adj[:, :, player_off] = p_int_adj[:, :, player_off]*(1-p_int_def[..., None])  # F, T, J
        p_int_off = 1-np.prod((1-p_int_adj[:, :, player_off]), axis=-1)  # F, T
        p_int_adj_torch = torchify(p_int_adj)

        p_int_off_only = np.minimum(1.0, 1-np.prod((1-p_int[:, :, player_off]), axis=-1))

        # projected locations at T (F, T, J)
        d_proj = np.select(
            [T[None, :, None] <= params.reax_t, T[None, :, None] <= (t_lt_smax + params.reax_t)[:, None, :], True],
            [0, (int_s0[:, None, :] * (T[None, :, None]-params.reax_t))+0.5*params.a_max*(T[None, :, None]-params.reax_t)**2,
             d_lt_smax[:, None, :] + params.s_max*(T[None, :, None] - t_lt_smax[:, None, :] - params.reax_t)])
        d_proj = np.minimum(d_proj, int_d_mag[:, None, :])  # no player is going to purposefully overshoot

        s_proj = np.select(
            [T[None, :, None] <= params.reax_t, T[None, :, None] <= (t_lt_smax + params.reax_t)[:, None, :],
             T[None, :, None] <= (t_lt_smax + t_at_smax + params.reax_t)[:, None, :],
             True],
            [int_s0[:, None, :],
             int_s0[:, None, :] + params.a_max * (T[None, :, None] - params.reax_t),
             params.s_max, params.s_max])

        x_proj = reaction_player_locs[None, None, :, 0] + d_proj*np.cos(int_d_theta[:, None, :])  # (F, T, J)
        y_proj = reaction_player_locs[None, None, :, 1] + d_proj*np.sin(int_d_theta[:, None, :])  # (F, T, J)

        v_x_proj = s_proj*np.cos(int_d_theta[:, None, :])  # (F, T, J)
        v_y_proj = s_proj*np.sin(int_d_theta[:, None, :])   # (F, T, J)

        # input: qb_loc (2,), t=frames_after_snap (int)
        # output: P(L,T|t) int probability of each pass (F, T)
        def get_hist_trans_prob():
            """ P(L|t) """
            ball_start_idx = np.rint(ball_start).astype(int)
            # mask for zeroing out parts of the field that are too far to be thrown to per the L_given_t model
            L_t_mask = np.zeros_like(xx, dtype=dt)  # (Y, X)
            L_t_mask[max(ball_start_idx[1]+y_min, 0):min(ball_start_idx[1]+y_max, len(y)-1),
                     max(ball_start_idx[0]+x_min, 0):min(ball_start_idx[0]+x_max, len(x)-1)] = 1.
            L_t_mask = L_t_mask.flatten()  # (F,)
            # # we clip reach vecs to be used to index into L_given_t.
            # # eg if qb is far right, then the left field will be clipped to y=-39 and later zeroed out
            # reach_vecs_int = np.rint(reach_vecs).astype(int)

            # clipped_reach_vecs = np.stack((np.clip(reach_vecs_int[:,0], x_min, x_max),
            #                               np.clip(-reach_vecs_int[:,1], y_min, y_max)))  # (2, F)
            # t_i = max(t-t_min, 0)
            # L_given_t = L_given_ts[t_i, clipped_reach_vecs[1]-y_min, clipped_reach_vecs[0]-x_min] * L_t_mask  # (F,) ; index with y and then x
            L_given_t = L_t_mask  # changed L_given_t to uniform after discussion
            L_given_t /= L_given_t.sum()  # renormalize since part of L|t may have been off field

            """ P(T|L) """
            # we find T|L for sufficiently close spots (1 < L <= 60)
            reach_dist_int = np.rint(np.linalg.norm(reach_vecs, axis=1)).astype(int)  # (F,)
            reach_dist_in_bounds_idx = (reach_dist_int > 1) & (reach_dist_int <= 60)
            reach_dist_in_bounds = reach_dist_int[reach_dist_in_bounds_idx]
            # yards 1-60 map to indices 0-59 so subtract 1 to get index
            # spots with dist 0 shouldn't wrap around to index -1 so clip
            T_given_L_subset = T_given_Ls[np.clip(reach_dist_in_bounds-1, 0, 59)]
            T_given_L = np.zeros((len(field_locs), len(T)), dtype=dt)  # (F, T)
            # fill in the subset of values computed above
            T_given_L[reach_dist_in_bounds_idx] = T_given_L_subset

            L_T_given_t = L_given_t[:, None] * T_given_L  # (F, T)
            L_T_given_t /= L_T_given_t.sum()  # normalize all passes after some have been chopped off
            return L_T_given_t

        def get_ppc():
            # use p_int as memoized values for integration

            # trajectory integration
            g = 10.72468  # y/s/s
            dx = reach_vecs_torch[:, 0]  # F
            dy = reach_vecs_torch[:, 1]  # F
            vx = dx[:, None]/T_torch[None, :]  # F, T
            vy = dy[:, None]/T_torch[None, :]  # F, T
            vz_0 = (T_torch*g)/2  # T

            # note that idx (i, j, k) into below arrays is invalid when j < k
            traj_ts = T_torch.repeat(field_locs_torch.shape[0], T_torch.shape[0], 1)  # (F, T, T)
            traj_locs_x_idx = torch.round(torch.clip((ball_start_torch[0]+vx[:, :, None]*T_torch), 0, len(x)-1)).long()  # F, T, T
            traj_locs_y_idx = torch.round(torch.clip((ball_start_torch[1]+vy[:, :, None]*T_torch), 0, len(y)-1)).long()  # F, T, T
            traj_locs_z = 2.0+vz_0[None, :, None]*traj_ts-0.5*g*traj_ts*traj_ts  # F, T, T
            path_idxs = (traj_locs_y_idx * xx.shape[1] + traj_locs_x_idx).flatten()  # (F*T*T,)
            traj_t_idxs = torch.round(10*traj_ts - 1).flatten().long()  # (F, T, T)
            ind_p_int_traj_dt = p_int_adj_torch[path_idxs, traj_t_idxs]
            ind_p_int_traj_dt = ind_p_int_traj_dt.reshape((*traj_locs_x_idx.shape), len(reaction_player_locs))
            # account for ball height on traj and normalize each locations int probability
            lambda_z = torch.where((traj_locs_z < params.z_max) & (traj_locs_z > params.z_min),
                                1, 0)  # F, T, T # maybe change this to a normal distribution
            ind_p_int_traj_dt = ind_p_int_traj_dt * lambda_z[:, :, :, None]
            # p_int_traj_sum = p_int_traj.sum(dim=-1)

            all_p_int_traj_dt = 1-torch.prod(1-ind_p_int_traj_dt, axis=-1)  # F, T, T

            # independent int probs at each point on trajectory
            # all_p_int_traj = torch.sum(p_int_traj_norm, dim=-1)  # F, T, T
            # off_p_int_traj = torch.sum(player_off_torch[None,None,None] * p_int_traj_norm, dim=-1)  # F, T, T
            # def_p_int_traj = torch.sum(np.logical_not(player_off_torch)[None,None,None] * p_int_traj_norm, dim=-1)  # F, T, T
            # ind_p_int_traj = p_int_traj_norm #use for analyzing specific players; # F, T, T, J
            # calc decaying residual probs after you take away p_int on earlier times in the traj
            compl_all_p_int_traj_dt = 1-all_p_int_traj_dt  # F, T, T
            remaining_compl_p_int_traj_dt = torch.cumprod(compl_all_p_int_traj_dt, dim=-1)  # F, T, T
            # maximum 0 because if it goes negative the pass has been caught by then and theres no residual probability
            shift_compl_cumsum_dt = torch.roll(remaining_compl_p_int_traj_dt, 1, dims=-1)  # F, T, T
            shift_compl_cumsum_dt[:, :, 0] = 1

            # multiply residual prob by p_int at that location
            # all_completion_prob_dt = shift_compl_cumsum * all_p_int_traj  # F, T, T
            # off_completion_prob_dt = shift_compl_cumsum * off_p_int_traj  # F, T, T
            # def_completion_prob_dt = shift_compl_cumsum * def_p_int_traj  # F, T, T
            ind_completion_prob_dt = shift_compl_cumsum_dt[:, :, :, None] * ind_p_int_traj_dt  # F, T, T, J

            # now accumulate values over total traj for each team and take at T=t
            # all_completion_prob = torch.cumsum(all_completion_prob_dt, dim=-1)  # F, T, T
            # off_completion_prob = torch.cumsum(off_completion_prob_dt, dim=-1)  # F, T, T
            # def_completion_prob = torch.cumsum(def_completion_prob_dt, dim=-1)  # F, T, T
            ind_completion_prob = torch.cumsum(ind_completion_prob_dt, dim=-2)  # F, T, T, J

            #     #### Toy example
            # all_p_int_traj = [0, 0, 0.1, 0.2, 0.8, 0.8]
            # c_all_p_int_traj=[1, 1, 0.9, 0.8, 0.2, 0.2]
            # rem_compl_p_int_traj = [1, 1, 0.9, 0.72, 0.144, 0.0288]
            # 0.1 + 0.9*0.2 + 0.72 * 0.8 + 0.144*0.8 = 0.9712
            # adjust_compl_prob =        [0, 0, 0.1, 0.28, 0.84, 0.84]

            # this einsum takes the diagonal values over the last two axes where T = t
            # this takes care of the t > T issue.
            # all_p_int_pass = np.einsum('ijj->ij', all_completion_prob)  # F, T
            # off_p_int_pass = torch.einsum('ijj->ij', off_completion_prob)  # F, T
            # def_p_int_pass = torch.einsum('ijj->ij', def_completion_prob)  # F, T
            ind_p_int_pass = torch.einsum('ijjk->ijk', ind_completion_prob)  # F, T, J
            # no_p_int_pass = 1-all_p_int_pass #F, T
            off_p_int_pass = torch.clip(torch.sum(ind_p_int_pass * player_off_torch[None, None], dim=-1), 0., 1.)  # F, T
            def_p_int_pass = torch.clip(torch.sum(ind_p_int_pass * player_def_torch[None, None], dim=-1), 0., 1.)  # F, T

            # assert np.allclose(all_p_int_pass, off_p_int_pass + def_p_int_pass, atol=0.01)
            # assert np.allclose(all_p_int_pass, ind_p_int_pass.sum(-1), atol=0.01)
            return off_p_int_pass, def_p_int_pass, ind_p_int_pass

        def get_xyac():
            x_proj_def = x_proj[:, :, player_def]  # F, T, J
            v_proj_def = s_proj[:, :, player_def]
            y_proj_def = y_proj[:, :, player_def]

            x_proj_relv = x_proj_def - field_locs[:, None, None, 0]  # F, T, J #GET RELATIVE COORDS
            y_proj_relv = y_proj_def - field_locs[:, None, None, 1]  # F, T, J #GET RELATIVE COORDS

            distances_to_ball = np.stack((x_proj_relv, y_proj_relv), axis=-1)  # F, T, J, 2
            distance_mags = np.linalg.norm(distances_to_ball, axis=-1)  # F, T, J

            sorted_indices = np.argsort(distance_mags, axis=2)  # F, T, J

            distance_mags = np.take_along_axis(distance_mags, sorted_indices, axis=2)  # F, T, J
            x_proj_sorted = np.take_along_axis(x_proj_relv, sorted_indices, axis=2)
            y_proj_sorted = np.take_along_axis(y_proj_relv, sorted_indices, axis=2)
            v_proj_sorted = np.take_along_axis(v_proj_def, sorted_indices, axis=2)

            just_top_5_distances = distance_mags[:, :, 0:5]  # F, T, 5
            just_top_5_x_proj = x_proj_sorted[:, :, 0:5]  # F, T, 5
            just_top_5_y_proj = y_proj_sorted[:, :, 0:5]  # F, T, 5
            just_top_5_v_proj = v_proj_sorted[:, :, 0:5]  # F, T, 5

            just_top_5_distances = np.reshape(just_top_5_distances, (-1, just_top_5_distances.shape[2]))  # (F*T, 5)
            just_top_5_x_proj = np.reshape(just_top_5_x_proj, just_top_5_distances.shape)
            just_top_5_y_proj = np.reshape(just_top_5_y_proj, just_top_5_distances.shape)
            just_top_5_v_proj = np.reshape(just_top_5_v_proj, just_top_5_distances.shape)

            endpoints = np.repeat(field_locs, repeats=len(T), axis=0)  # FxT, 2
            # assert((field_locs[:, None, :]+np.zeros_like(T[None, None, :])).reshape(end))
            times = np.repeat(T[None, :], repeats=len(field_locs), axis=0)
            times_shaped = times.reshape((times.shape[0]*times.shape[1]))  # FxT, 1
            value_array = np.array([-2.5, 2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 30])

            field_df = pd.DataFrame({
                'pass_endpoint_x': endpoints[:, 0],
                'pass_endpoint_y': endpoints[:, 1],
                'frame_thrown': t,
                'time_of_flight': times_shaped,
                '1-closest-defender-distance': just_top_5_distances[:, 0],
                '2-closest-defender-distance': just_top_5_distances[:, 1],
                '3-closest-defender-distance': just_top_5_distances[:, 2],
                '4-closest-defender-distance': just_top_5_distances[:, 3],
                '5-closest-defender-distance': just_top_5_distances[:, 4],
                '1-closest-defender-x': just_top_5_x_proj[:, 0],
                '2-closest-defender-x': just_top_5_x_proj[:, 1],
                '3-closest-defender-x': just_top_5_x_proj[:, 2],
                '4-closest-defender-x': just_top_5_x_proj[:, 3],
                '5-closest-defender-x': just_top_5_x_proj[:, 4],
                '1-closest-defender-y': just_top_5_y_proj[:, 0],
                '2-closest-defender-y': just_top_5_y_proj[:, 1],
                '3-closest-defender-y': just_top_5_y_proj[:, 2],
                '4-closest-defender-y': just_top_5_y_proj[:, 3],
                '5-closest-defender-y': just_top_5_y_proj[:, 4],
                '1-closest-defender-speed': just_top_5_v_proj[:, 0],
                '2-closest-defender-speed': just_top_5_v_proj[:, 1],
                '3-closest-defender-speed': just_top_5_v_proj[:, 2],
                '4-closest-defender-speed': just_top_5_v_proj[:, 3],
                '5-closest-defender-speed': just_top_5_v_proj[:, 4],
                "y": endpoints[:, 1]

            })
            # CALCULTE XYAC

            dtest = treelite_runtime.Batch.from_npy2d(field_df[cols_when_model_builds].values)
            ypred = xyac_predictor.predict(dtest)
            y_vals = np.sum(ypred*value_array, axis=1)
            field_df['xyac'] = y_vals
            field_df['play_endpoint_x'] = np.round(field_df['xyac'] + field_df['pass_endpoint_x'])
            field_df['play_endpoint_x'] = field_df['play_endpoint_x']+.5
            field_df['play_endpoint_x'] = np.clip(field_df['play_endpoint_x'], 0.5, 119.5)

            return field_df[['pass_endpoint_x', 'pass_endpoint_y', 'time_of_flight', 'xyac', 'play_endpoint_x']]

        nonlocal epa_df

        ppc_off, ppc_def, ppc_ind = get_ppc()  # (F, T), (F, T), (F, T, J)
        ppc_off, ppc_def, ppc_ind = ppc_off.detach().cpu().numpy(), ppc_def.detach().cpu().numpy(), ppc_ind.detach().cpu().numpy()
        ind_info = np.stack((player_ids, player_teams), axis=1)
        h_trans_prob = get_hist_trans_prob()

        epa_xyac_df = get_xyac().merge(epa_df, how='left', on='play_endpoint_x')
        xyac = epa_xyac_df.xyac.to_numpy().reshape(ppc_off.shape)
        # end_x = epa_xyac_df.play_endpoint_x.to_numpy().reshape(ppc_off.shape)
        xepa = epa_xyac_df.xepa.to_numpy().reshape(ppc_off.shape)  # F, T
        # assert(h_trans_prob.shape == ppc_off.shape)

        ppc = ppc_off

        trans_ppc = h_trans_prob * np.power(ppc, params.alpha)  # F, T
        trans_pint = h_trans_prob * np.power(p_int_off_only, params.alpha)  # F, T
        trans_ppc /= trans_ppc.sum()
        trans_pint /= trans_pint.sum()

        eppa_wo_value_ppc_ind = ppc_ind * trans_ppc[..., None]  # F, T, J
        eppa_wo_value_pint_ind = ppc_ind * trans_pint[..., None]  # F, T, J
        eppa_ppc_ind = eppa_wo_value_ppc_ind*xepa[..., None]  # F, T, J
        eppa_pint_ind = eppa_wo_value_pint_ind*xepa[..., None]  # F, T, J
        eppa_ppc = np.sum(eppa_ppc_ind, axis=-1)  # F, T
        eppa_pint = np.sum(eppa_pint_ind, axis=-1)  # F, T
        eppa3 = p_int_off*trans_pint*xepa
        eppa4 = p_int_off*h_trans_prob * np.power(p_int_off, params.alpha)*xepa
        eppa_wo_trans = ppc*xepa  # F, T
        eppa_wo_value_ppc = np.sum(eppa_wo_value_ppc_ind, axis=-1)
        eppa_wo_value_pint = np.sum(eppa_wo_value_pint_ind, axis=-1)

        field_df = pd.DataFrame()
        player_stats_df = pd.DataFrame()
        passes_df = pd.DataFrame()
        proj_df = pd.DataFrame()

        if viz_true_proj and true_pass_found:
            proj_df['nflId'] = frame_df['nflId']
            proj_df['reax_x'] = frame_df['x_r']
            proj_df['reax_y'] = frame_df['y_r']
            proj_df['reax_v_x'] = frame_df['v_x_r']
            proj_df['reax_v_y'] = frame_df['v_y_r']

            proj_df['d_vec_x'] = int_d_vec[true_f_idx, :, 0]
            proj_df['d_vec_y'] = int_d_vec[true_f_idx, :, 1]
            proj_df['d_mag'] = int_d_mag[true_f_idx]
            proj_df['int_s0'] = int_s0[true_f_idx]

            proj_df['t_lt_smax'] = t_lt_smax[true_f_idx]
            proj_df['d_lt_smax'] = d_lt_smax[true_f_idx]
            proj_df['t_at_smax'] = t_at_smax[true_f_idx]
            proj_df['d_at_smax'] = d_at_smax[true_f_idx]
            proj_df['t_tot'] = t_tot[true_f_idx]

            proj_df['int_dT'] = int_dT[true_f_idx, true_T_idx]
            proj_df['p_int'] = p_int[true_f_idx, true_T_idx]
            proj_df['p_int_adj'] = p_int_adj[true_f_idx, true_T_idx]

            proj_df['d_proj'] = d_proj[true_f_idx, true_T_idx]  # J
            proj_df['s_proj'] = s_proj[true_f_idx, true_T_idx]  # J

            proj_df['proj_x'] = x_proj[true_f_idx, true_T_idx]  # J
            proj_df['proj_y'] = y_proj[true_f_idx, true_T_idx]  # J
            proj_df['proj_v_x'] = v_x_proj[true_f_idx, true_T_idx]  # J
            proj_df['proj_v_y'] = v_y_proj[true_f_idx, true_T_idx]  # J

            proj_df['ppc_ind'] = ppc_ind[true_f_idx, true_T_idx]

            proj_df['frameId'] = frame_id

        if save_np:
            week = games_df.loc[games_df.gameId == game_id].week.to_numpy()[0].item()
            dir = out_dir_path.format(f'{week}/{game_id}/{play_id}')
            Path(dir).mkdir(parents=True, exist_ok=True)
            # np.savez_compressed(f'{dir}/{frame_id}', players_ind_info=ind_info,
            #                     ppc_ind=ppc_ind, h_trans=h_trans_prob, xepa=xepa, eppa_ind=eppa_ind)
            np.savez_compressed(f'{dir}/{frame_id}', ind_info=ind_info, eppa_ind=eppa_ppc_ind)

        if stats_df:
            ind_eppa1 = eppa_ppc_ind.sum(axis=(0, 1))  # J
            ind_eppa2 = eppa_pint_ind.sum(axis=(0, 1))  # J
            ind_eppa1_wo_value = eppa_wo_value_ppc_ind.sum(axis=(0, 1))  # J
            ind_eppa2_wo_value = eppa_wo_value_pint_ind.sum(axis=(0, 1))  # J
            player_stats_df = pd.DataFrame(
                {'gameId': game_id, 'playId': play_id, 'frameId': frame_id, 'frame_after_snap': t, 'nflId': player_ids,
                 'displayName': player_names, 'team': player_team_names, 'team_pos': player_teams,
                 'ind_eppa1': ind_eppa1, 'ind_eppa2': ind_eppa2, 'ind_eppa1_wo_value': ind_eppa1_wo_value, 'ind_eppa2_wo_value': ind_eppa2_wo_value},
                index=player_ids)

            off_team_name = play_df.loc[play_df.team_pos == 'OFF'].teamAbbr.iloc[0]
            def_team_name = play_df.loc[play_df.team_pos == 'DEF'].teamAbbr.iloc[0]

            row = {'gameId': game_id, 'playId': play_id, 'frameId': frame_id, 'frames_after_snap': t, 'off_team': off_team_name,
                   'def_team': def_team_name, 'eppa1_tot': eppa_ppc.sum(),
                   'eppa2_tot': eppa_pint.sum(),
                   'eppa3_tot': eppa3.sum(),
                   'eppa4_tot': eppa4.sum(),
                   'eppa1_wo_value': eppa_wo_value_ppc.sum(),
                   'eppa2_wo_value': eppa_wo_value_pint.sum(),
                   'eppa1_wo_trans': eppa_wo_trans.mean()}

            for name, arr in {'eppa1': eppa_ppc, 'eppa2': eppa_pint, 'ppc_off': ppc_off, 'ppc_def': ppc_def, 'eppa_wo_trans': eppa_wo_trans,
                              'eppa1_wo_value': eppa_wo_value_ppc, 'eppa2_wo_value': eppa_wo_value_pint}.items():
                f, T_idx = np.unravel_index(arr.argmax(), arr.shape)
                end_x, end_y = field_locs[f]
                row[f"max_{name}_x"] = end_x
                row[f"max_{name}_y"] = end_y
                row[f"max_{name}_T"] = T[T_idx]
                row[f"max_{name}_ppc_off"] = ppc_off[f, T_idx]
                row[f"max_{name}_pint_off"] = p_int_off[f, T_idx]
                row[f"max_{name}_ppc_def"] = ppc_def[f, T_idx]
                row[f"max_{name}_pint_def"] = p_int_def[f, T_idx]
                row[f"max_{name}_xyac"] = xyac[f, T_idx]
                row[f"max_{name}_xepa"] = xepa[f, T_idx]
                row[f"max_{name}_trans_ppc"] = trans_ppc[f, T_idx]
                row[f"max_{name}_trans_pint"] = trans_pint[f, T_idx]
                row[f"max_{name}_trans_tof"] = h_trans_prob[f, T_idx]
                row[f"max_{name}_eppa1"] = eppa_ppc[f, T_idx]
                row[f"max_{name}_eppa2"] = eppa_pint[f, T_idx]
                row[f"max_{name}_eppa_wo_trans"] = eppa_wo_trans[f, T_idx]
                row[f"max_{name}_eppa1_wo_value"] = eppa_wo_value_ppc[f, T_idx]
                row[f"max_{name}_eppa2_wo_value"] = eppa_wo_value_pint[f, T_idx]

            if frame_id == pass_forward_frame and true_pass_found:
                row[f"true_x"] = true_x_idxd
                row[f"true_y"] = true_y_idxd
                row[f"true_T"] = true_T_idxd
                row[f"true_ppc_off"] = ppc_off[true_f_idx, true_T_idx]
                row[f"true_pint_off"] = p_int_off[true_f_idx, true_T_idx]
                row[f"true_ppc_def"] = ppc_def[true_f_idx, true_T_idx]
                row[f"true_pint_def"] = p_int_def[true_f_idx, true_T_idx]
                row[f"true_xyac"] = xyac[true_f_idx, true_T_idx]
                row[f"true_xepa"] = xepa[true_f_idx, true_T_idx]
                row[f"true_trans_ppc"] = trans_ppc[true_f_idx, true_T_idx]
                row[f"true_trans_pint"] = trans_pint[true_f_idx, true_T_idx]
                row[f"true_trans_tof"] = h_trans_prob[true_f_idx, true_T_idx]
                row[f"true_eppa1"] = eppa_ppc[true_f_idx, true_T_idx]
                row[f"true_eppa2"] = eppa_pint[true_f_idx, true_T_idx]
                row[f"true_eppa_wo_trans"] = eppa_wo_trans[true_f_idx, true_T_idx]
                row[f"true_eppa1_wo_value"] = eppa_wo_value_ppc[true_f_idx, true_T_idx]
                row[f"true_eppa2_wo_value"] = eppa_wo_value_pint[true_f_idx, true_T_idx]

            passes_df = pd.DataFrame(row, index=[t])

        if viz_df:
            field_df = pd.DataFrame({
                'frameId': frame_id,
                'ball_end_x': field_locs[:, 0],
                'ball_end_y': field_locs[:, 1],
                'eppa1': eppa_ppc.max(axis=1),
                'eppa2': eppa_pint.max(axis=1),
                'eppa3': eppa3.max(axis=1),
                'eppa4': eppa3.max(axis=1),
                'p_int_off_only': p_int_off_only.max(axis=1),
                'p_int_off': p_int_off.max(axis=1),
                # 'p_int_adj_off': np.max(p_int_adj, axis=(1, 2), where=(player_teams=='OFF')),
                'ppc_off': ppc_off.max(axis=1),
                'ppc_def': ppc_def.max(axis=1),
                'eppa1_wo_value': eppa_wo_value_ppc.max(axis=1),
                'eppa2_wo_value': eppa_wo_value_pint.max(axis=1),
                'eppa_wo_trans': eppa_wo_trans.max(axis=1),
                'xyac': xyac.max(axis=1),
                'xepa': xepa.max(axis=1),
            })
            field_df.loc[field_df.ball_end_x < ball_start[0]-10, :] = np.nan  # remove backward passes
        # breakpoint()
        return field_df, passes_df, player_stats_df, proj_df

    field_dfs = pd.DataFrame()
    player_stats_df = pd.DataFrame()
    passes_df = pd.DataFrame()
    proj_df = pd.DataFrame()
    #for fid in tqdm(range(ball_snap_frame+1, pass_forward_frame+1)):
    for fid in tqdm(range(pass_forward_frame-5, pass_forward_frame+1)):
        field_df, passes, player_stats, projs = frame_eppa(fid)
        field_dfs = field_dfs.append(field_df, ignore_index=True)
        passes_df = passes_df.append(passes, ignore_index=True)
        player_stats_df = player_stats_df.append(player_stats, ignore_index=True)
        proj_df = proj_df.append(projs, ignore_index=True)

    if true_pass_found & viz_true_proj:
        play_df = pd.merge(play_df, proj_df, on=['frameId', 'nflId'], how='left')

    if save_all_dfs:
        week = games_df.loc[games_df.gameId == game_id].week.to_numpy()[0].item()
        dir = out_dir_path.format(f'{week}/{game_id}/{play_id}')
        Path(dir).mkdir(parents=True, exist_ok=True)
        proj_df.to_pickle(f"{dir}/true_pass_player_proj.pkl",)
        passes_df.to_pickle(f"{dir}/passes.pkl",)
        player_stats_df.to_pickle(f"{dir}/player_stats.pkl",)
        field_dfs.to_pickle(f"{dir}/field_viz.pkl",)
    return play_df, field_dfs, passes_df, player_stats_df


plays = sorted(list(set(map(lambda x: (x[0].item(), x[1].item()), track_df.groupby(
    ['gameId', 'playId'], as_index=False).first()[['gameId', 'playId']].to_numpy()))))


# plays = [(2018102806, 2425)]

fails = []
Path(out_dir_path.format(f'{WEEK}')).mkdir(parents=True, exist_ok=True)
with open(out_dir_path.format(f'{WEEK}/errors.txt'), 'w+') as f:
    # for (gid, pid) in tqdm(random.sample(plays, len(plays))):
    for (gid, pid) in tqdm(plays[:3]):
        dir = out_dir_path.format(f'{WEEK}/{gid}/{pid}')
        if os.path.exists(dir) and False:
            print(f'EXISTS: {gid}, {pid}')
        else:
            try:
                play_eppa(gid, pid, viz_df=False, save_np=False, stats_df=True, viz_true_proj=True, save_all_dfs=False)
            except Exception as e:
                fails.append((gid, pid))
                f.write(f"\nERROR: {gid}, {pid}\n")
                f.write(traceback.format_exc()+'\n\n\n')
                print(f"ERROR: {gid}, {pid}. {e}")
    print(len(plays))
    print(len(fails))
    f.write(f"{len(fails)} out of {len(plays)}")
    f.write(str(fails))

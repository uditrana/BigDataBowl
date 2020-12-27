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

# file loading and prep
path_shared = '~/Downloads/nfl-big-data-bowl-2021/{}'

games_df = pd.read_csv(path_shared.format('games.csv'))
plays_df = pd.read_csv(path_shared.format('plays.csv'))
players_df = pd.read_csv(path_shared.format('players.csv'))
track_df = pd.read_csv(
    path_shared.format(
        'week1_norm.csv',
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
bst = joblib.load("./in/xyac_model.model")
xgb.plot_importance(bst)
scores = bst.get_score(importance_type='gain')
print(scores.keys())
cols_when_model_builds = bst.feature_names
model = treelite.Model.from_xgboost(bst)
toolchain = 'gcc'
model.export_lib(toolchain=toolchain, libpath='./in/xyacmymodel.so',
                 params={'parallel_comp': 32}, verbose=True)  # .so for ubuntu, .dylib for mac

bst = joblib.load("./in/epa_model_rishav_no_time.model")
xgb.plot_importance(bst)
scores = bst.get_score(importance_type='gain')
print(scores.keys())
cols_when_model_builds = bst.feature_names
model = treelite.Model.from_xgboost(bst)
toolchain = 'gcc'
model.export_lib(toolchain=toolchain, libpath='./in/epa_no_time_mymodel.so',
                 params={'parallel_comp': 32}, verbose=True)  # .so for ubuntu, .dylib for mac


def params(): return None  # create an empty object to add params


params.a_max = 7
params.s_max = 9
# params.reax_t = params.s_max/params.a_max
params.reax_t = 0.2
params.tti_sigma = 0.318
# params.tti_sigma = 0.45
params.cell_length = 1
params.alpha = 1.5
params.z_min = 1
params.z_max = 3
vars(params)

dt = np.float64

# model constants
T = np.linspace(0.1, 4, 40, dtype=dt)
x = np.linspace(0.5, 119.5, 120, dtype=dt)
y = np.linspace(-0.5, 53.5, 55, dtype=dt)
y[0] = -0.2
xx, yy = np.meshgrid(x, y)
field_locs = np.stack((xx, yy)).reshape(2, -1).T  # (F, 2)
tot_pass_cnt = len(field_locs[:, 1])*len(T)
print(f'Considering {tot_pass_cnt} passes per frame')

# historical trans model inputs/params
L_given_ts = np.load('in/L_given_t.npy')
T_given_Ls_df = pd.read_pickle('in/T_given_L.pkl')
# from L_given_t in historical notebook
x_min, x_max = -9, 70
y_min, y_max = -39, 40
t_min, t_max = 10, 63

# epa/xyac model loading
bst = joblib.load("./in/xyac_model.model")
scores = bst.get_score(importance_type='gain')
cols_when_model_builds = bst.feature_names
xyac_predictor = treelite_runtime.Predictor('./in/xyacmymodel.dylib')
epa_model = joblib.load("./in/epa_model_rishav_no_time.model")
scores = epa_model.get_score(importance_type='gain')
cols_when_model_builds_ep = epa_model.feature_names
epa_predictor = treelite_runtime.Predictor('./in/epa_no_time_mymodel.dylib')


def play_eppa(game_id, play_id, viz_df=False, save_np=False, stats_df=False, viz_true_proj=False, save_all_dfs=False):
    play_df = track_df[(track_df.playId == play_id) & (track_df.gameId == game_id)].sort_values(by='frameId')

    ball_snap_frame = play_df.loc[(play_df.nflId == 0) & (play_df.event == 'ball_snap')].frameId.iloc[0]
    pass_forward_frame = play_df.loc[(play_df.nflId == 0) & (play_df.event == 'pass_forward')].frameId.iloc[0]
    play_df['frames_since_snap'] = play_df.frameId - ball_snap_frame

    pass_arrived_frame = play_df.loc[(play_df.nflId == 0) & (play_df.event == 'pass_arrived')].frameId.iloc[0]
    true_T_frames = pass_arrived_frame - pass_forward_frame
    true_x, true_y = play_df.loc[(play_df.nflId == 0) & (play_df.event == 'pass_arrived'),
                                 ['x', 'y']].iloc[0].to_numpy(dtype=dt)
    print(f"True Pass: t: {pass_forward_frame} x:{true_x} y:{true_y} T:{true_T_frames/10}")
    true_T_ind = np.rint(true_T_frames).astype(int)-1
    true_x_ind = (true_x).astype(int)
    true_y_ind = (true_y).astype(int)+1
    true_pass_f_ind = np.ravel_multi_index((true_y_ind, true_x_ind), xx.shape)
    x_idxd, y_idxd = field_locs[true_pass_f_ind]
    print(f"True Pass idxd: t: {pass_forward_frame} x:{x_idxd} y:{y_idxd} T:{T[true_T_ind]}")

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
        t = frame_df.frames_since_snap.iloc[0]
        frame_df = frame_df.loc[(frame_df.nflId != 0) & (frame_df.position != 'QB')]  # remove ball and qb from df

        reach_vecs = (field_locs - ball_start).astype(dt)  # (F, 2)

        frame_df = frame_df.drop_duplicates(subset='nflId').sort_values('nflId').reset_index()

        # project motion by reaction time
        frame_df['v_x_r'] = frame_df.a_x*params.reax_t+frame_df.v_x
        frame_df['v_y_r'] = frame_df.a_y*params.reax_t+frame_df.v_y
        frame_df['v_r_mag'] = np.linalg.norm(np.array([frame_df.v_x_r, frame_df.v_y_r], dtype=dt), axis=0)
        frame_df['v_r_theta'] = np.arctan2(frame_df.v_y_r, frame_df.v_x_r)
        frame_df['x_r'] = frame_df.x + frame_df.v_x*params.reax_t + 0.5*frame_df.a_x*params.reax_t**2
        frame_df['y_r'] = frame_df.y + frame_df.v_y*params.reax_t + 0.5*frame_df.a_y*params.reax_t**2

        player_teams = frame_df['team_pos'].to_numpy()  # J,
        player_team_names = frame_df['teamAbbr'].to_numpy()  # J,
        player_ids = frame_df['nflId'].to_numpy()
        player_names = frame_df['displayName'].to_numpy()
        reaction_player_locs = frame_df[['x_r', 'y_r']].to_numpy(dtype=dt)  # (J, 2)
        reaction_player_vels = frame_df[['v_x_r', 'v_y_r']].to_numpy(dtype=dt)  # (J,2)

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
        int_dT = T[None, :, None] - t_tot[:, None, :]  # F, T, J
        p_int = (1/(1. + np.exp(-np.pi/np.sqrt(3.0)/params.tti_sigma * int_dT, dtype=dt)))  # F, T, J

        # projected locations at T (F, T, J)
        d_proj = np.select(
            [T[None, :, None] <= params.reax_t, T[None, :, None] <= (t_lt_smax + params.reax_t)[:, None, :],
             T[None, :, None] <= (t_lt_smax + t_at_smax + params.reax_t)[:, None, :],
             True],
            [0, (int_s0[:, None, :] * (T[None, :, None]-params.reax_t))+0.5*params.a_max*(T[None, :, None]-params.reax_t)**2,
             d_lt_smax[:, None, :] + d_at_smax[:, None, :] * np.true_divide(
                 (T[None, :, None] - t_lt_smax[:, None, :] - params.reax_t),
                 t_at_smax[:, None, :],
                 out=np.zeros_like(int_dT),
                 where=t_at_smax[:, None, :] != 0),
             int_d_mag[:, None, :]])
        d_proj = np.clip(d_proj, 0, int_d_mag[:, None, :])  # no player is going to purposefully overshoot

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

        proj_df = pd.DataFrame()
        if viz_true_proj:
            proj_df['nflId'] = frame_df['nflId']
            proj_df['reax_x'] = frame_df['x_r']
            proj_df['reax_y'] = frame_df['y_r']
            proj_df['reax_v_x'] = frame_df['v_x_r']
            proj_df['reax_v_y'] = frame_df['v_y_r']

            proj_df['proj_x'] = x_proj[true_pass_f_ind, true_T_ind]  # J
            proj_df['proj_y'] = y_proj[true_pass_f_ind, true_T_ind]  # J
            proj_df['proj_v_x'] = v_x_proj[true_pass_f_ind, true_T_ind]  # J
            proj_df['proj_v_y'] = v_y_proj[true_pass_f_ind, true_T_ind]  # J

            proj_df['int_dT'] = int_dT[true_pass_f_ind, true_T_ind]
            proj_df['p_int'] = p_int[true_pass_f_ind, true_T_ind]
            proj_df['d_vec_x'] = int_d_vec[true_pass_f_ind, :, 0]
            proj_df['d_vec_y'] = int_d_vec[true_pass_f_ind, :, 1]
            proj_df['d_mag'] = int_d_mag[true_pass_f_ind]
            proj_df['int_s0'] = int_s0[true_pass_f_ind]
            proj_df['t_lt_smax'] = t_lt_smax[true_pass_f_ind]
            proj_df['d_lt_smax'] = d_lt_smax[true_pass_f_ind]
            proj_df['t_at_smax'] = t_at_smax[true_pass_f_ind]
            proj_df['d_at_smax'] = d_at_smax[true_pass_f_ind]

            proj_df['t_tot'] = t_tot[true_pass_f_ind]
            proj_df['d_proj'] = d_proj[true_pass_f_ind, true_T_ind]  # J
            proj_df['s_proj'] = s_proj[true_pass_f_ind, true_T_ind]  # J

            proj_df['frameId'] = frame_id

        # input: qb_loc (2,), t=frames_after_snap (int)
        # output: (P(L,T)|t) int probability of each pass (F, T)

        def get_hist_trans_prob():
            """ P(L|t) """
            ball_start_ind = np.rint(ball_start).astype(int)
            # mask for zeroing out parts of the field that are too far to be thrown to per the L_given_t model
            L_t_mask = np.zeros_like(xx, dtype=dt)  # (Y, X)
            L_t_mask[max(ball_start_ind[1]+y_min, 0):min(ball_start_ind[1]+y_max, len(y)-1),
                     max(ball_start_ind[0]+x_min, 0):min(ball_start_ind[0]+x_max, len(x)-1)] = 1.
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
            T_given_L_subset = T_given_Ls_df.set_index('dist').loc[reach_dist_in_bounds, 'p'].to_numpy(dtype=dt)\
                .reshape(len(reach_dist_in_bounds), -1)  # (F~, T) ; F~ is subset of F that is in [1, 60] yds from ball
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
            dx = reach_vecs[:, 0]  # F
            dy = reach_vecs[:, 1]  # F
            vx = dx[:, None]/T[None, :]  # F, T
            vy = dy[:, None]/T[None, :]  # F, T
            vz_0 = (T*g)/2  # T

            # note that idx (i, j, k) into below arrays is invalid when j < k
            traj_ts = np.tile(T, (len(field_locs), len(T), 1))  # (F, T, T)
            traj_locs_x_idx = np.rint(np.clip((ball_start[0]+vx[:, :, None]*T), 0, len(x)-1)).astype(int)  # F, T, T
            traj_locs_y_idx = np.rint(np.clip((ball_start[1]+vy[:, :, None]*T), 0, len(y)-1)).astype(int)  # F, T, T
            traj_locs_z = 2.0+vz_0[None, :, None]*traj_ts-0.5*g*traj_ts*traj_ts  # F, T, T
            path_idxs = np.ravel_multi_index(
                np.stack((traj_locs_y_idx, traj_locs_x_idx)).reshape(2, -1),
                xx.shape)  # (F*T*T,)
            traj_t_idxs = np.rint(10*traj_ts - 1).flatten().astype(int)  # (F, T, T)
            p_int_traj = p_int[path_idxs, traj_t_idxs]  # F*T*T, J
            p_int_traj = p_int_traj.reshape((*traj_locs_x_idx.shape, len(reaction_player_locs)))  # F, T, T, J

            # account for ball height on traj and normalize each locations int probability
            lambda_z = np.where((traj_locs_z < params.z_max) & (traj_locs_z > params.z_min),
                                1, 0)  # F, T, T # maybe change this to a normal distribution
            p_int_traj = p_int_traj * lambda_z[:, :, :, None]
            norm_factor = np.maximum(1., p_int_traj.sum(axis=-1))  # F, T, T
            p_int_traj_norm = (p_int_traj/norm_factor[..., None])  # F, T, T, J

            # independent int probs at each point on trajectory
            all_p_int_traj = np.sum(p_int_traj_norm, axis=-1)  # F, T, T
            off_p_int_traj = np.sum(p_int_traj_norm, axis=-1, where=(player_teams == 'OFF'))
            def_p_int_traj = np.sum(p_int_traj_norm, axis=-1, where=(player_teams == 'DEF'))
            ind_p_int_traj = p_int_traj_norm  # use for analyzing specific players

            # calc decaying residual probs after you take away p_int on earlier times in the traj
            compl_all_p_int_traj = 1-all_p_int_traj  # F, T, T
            remaining_compl_p_int_traj = np.cumprod(compl_all_p_int_traj, axis=-1)  # F, T, T
            # maximum 0 because if it goes negative the pass has been caught by then and theres no residual probability
            shift_compl_cumsum = np.roll(remaining_compl_p_int_traj, 1, axis=-1)  # F, T, T
            shift_compl_cumsum[:, :, 0] = 1

            # multiply residual prob by p_int at that location
            # all_completion_prob_dt = shift_compl_cumsum * all_p_int_traj  # F, T, T
            off_completion_prob_dt = shift_compl_cumsum * off_p_int_traj  # F, T, T
            def_completion_prob_dt = shift_compl_cumsum * def_p_int_traj  # F, T, T
            ind_completion_prob_dt = shift_compl_cumsum[:, :, :, None] * ind_p_int_traj  # F, T, T, J

            # now accumulate values over total traj for each team and take at T=t
            # all_completion_prob = np.cumsum(all_completion_prob_dt, axis=-1)  # F, T, T
            off_completion_prob = np.cumsum(off_completion_prob_dt, axis=-1)  # F, T, T
            def_completion_prob = np.cumsum(def_completion_prob_dt, axis=-1)  # F, T, T
            ind_completion_prob = np.cumsum(ind_completion_prob_dt, axis=-2)  # F, T, T, J

            #     #### Toy example
            # all_p_int_traj = [0, 0, 0.1, 0.2, 0.8, 0.8]
            # c_all_p_int_traj=[1, 1, 0.9, 0.8, 0.2, 0.2]
            # rem_compl_p_int_traj = [1, 1, 0.9, 0.72, 0.144, 0.0288]
            # 0.1 + 0.9*0.2 + 0.72 * 0.8 + 0.144*0.8 = 0.9712
            # adjust_compl_prob =        [0, 0, 0.1, 0.28, 0.84, 0.84]

            # this einsum takes the diagonal values over the last two axes where T = t
            # this takes care of the t > T issue.
            # all_p_int_pass = np.einsum('ijj->ij', all_completion_prob)  # F, T
            off_p_int_pass = np.einsum('ijj->ij', off_completion_prob)  # F, T
            def_p_int_pass = np.einsum('ijj->ij', def_completion_prob)  # F, T
            ind_p_int_pass = np.einsum('ijjk->ijk', ind_completion_prob)  # F, T, J
            # no_p_int_pass = 1-all_p_int_pass #F, T

            # assert np.allclose(all_p_int_pass, off_p_int_pass + def_p_int_pass, atol=0.01)
            # assert np.allclose(all_p_int_pass, ind_p_int_pass.sum(-1), atol=0.01)
            return off_p_int_pass, def_p_int_pass, ind_p_int_pass

        def get_xyac():
            x_proj_def = x_proj[:, :, player_teams == "DEF"]  # F, T, J
            v_proj_def = s_proj[:, :, player_teams == "DEF"]
            y_proj_def = y_proj[:, :, player_teams == "DEF"]

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
        ind_info = np.stack((player_ids, player_teams), axis=1)
        h_trans_prob = get_hist_trans_prob()

        epa_xyac_df = get_xyac().merge(epa_df, how='left', on='play_endpoint_x')
        xyac = epa_xyac_df.xyac.to_numpy().reshape(ppc_off.shape)
        # end_x = epa_xyac_df.play_endpoint_x.to_numpy().reshape(ppc_off.shape)
        xepa = epa_xyac_df.xepa.to_numpy().reshape(ppc_off.shape)  # F, T
        # assert(h_trans_prob.shape == ppc_off.shape)

        ppc = ppc_off
        trans_prob = h_trans_prob * np.power(ppc, params.alpha)  # F, T
        trans_prob /= trans_prob.sum()

        ppc_trans_ind = ppc_ind * trans_prob[..., None]  # F, J, T
        eppa_ind = ppc_trans_ind*xepa[..., None]  # F, J, T
        eppa = ppc*trans_prob*xepa  # F, T
        ppc_value = ppc*xepa  # F, T
        ppc_trans = ppc*trans_prob  # F, T

        # TORCH UP TO HERE!

        field_df = pd.DataFrame()
        player_stats_df = pd.DataFrame()
        passes_df = pd.DataFrame()

        if save_np:
            week = games_df.loc[games_df.gameId == game_id].week.to_numpy()[0].item()
            dir = out_dir_path.format(f'{week}/{game_id}/{play_id}')
            Path(dir).mkdir(parents=True, exist_ok=True)
            # np.savez_compressed(f'{dir}/{frame_id}', players_ind_info=ind_info,
            #                     ppc_ind=ppc_ind, h_trans=h_trans_prob, xepa=xepa, eppa_ind=eppa_ind)
            np.savez_compressed(f'{dir}/{frame_id}', ind_info=ind_info, eppa_ind=eppa_ind)

        if stats_df:
            ind_eppa_sum = eppa_ind.sum(axis=(0, 1))  # J
            ppc_trans_ind_sum = ppc_trans_ind.sum(axis=(0, 1))  # J
            player_stats_df = pd.DataFrame(
                {'gameId': game_id, 'playId': play_id, 'frameId': frame_id, 'frame_after_snap': t, 'nflId': player_ids,
                 'displayName': player_names, 'team': player_team_names, 'team_pos': player_teams,
                 'eppa_ind': ind_eppa_sum, 'ppc_trans': ppc_trans_ind_sum},
                index=player_ids)

            off_team_name = play_df.loc[play_df.team_pos == 'OFF'].teamAbbr.iloc[0]
            def_team_name = play_df.loc[play_df.team_pos == 'DEF'].teamAbbr.iloc[0]

            row = {'gameId': game_id, 'playId': play_id, 'frameId': frame_id, 'frames_after_snap': t,
                   'off_team': off_team_name, 'def_team': def_team_name, 'eppa_tot': eppa.sum(),
                   'ppc_trans_tot': ppc_trans.sum()}
            for name, arr in {'EPPA': eppa, 'PPC': ppc_off, 'PPC_DEF': ppc_def, 'PPCbyVALUE': ppc_value,
                              'PPCbyTRANS': ppc_trans}.items():
                f, T_ind = np.unravel_index(arr.argmax(), arr.shape)
                end_x, end_y = field_locs[f]
                row[f"max{name}_x"] = end_x
                row[f"max{name}_y"] = end_y
                row[f"max{name}_T"] = T[T_ind]
                row[f"max{name}_ppc_off"] = ppc_off[f, T_ind]
                row[f"max{name}_ppc_def"] = ppc_def[f, T_ind]
                row[f"max{name}_xyac"] = xyac[f, T_ind]
                row[f"max{name}_xepa"] = xepa[f, T_ind]
                row[f"max{name}_trans"] = trans_prob[f, T_ind]
                row[f"max{name}_trans_denorm"] = trans_prob[f, T_ind]*tot_pass_cnt
                row[f"max{name}_hist_trans"] = h_trans_prob[f, T_ind]
                row[f"max{name}_hist_trans_denorm"] = h_trans_prob[f, T_ind]*tot_pass_cnt
                row[f"max{name}_eppa"] = eppa[f, T_ind]
                row[f"max{name}_eppa_denorm"] = eppa[f, T_ind]*tot_pass_cnt
                row[f"max{name}_ppc_val"] = ppc_value[f, T_ind]
                row[f"max{name}_ppc_trans"] = ppc_trans[f, T_ind]
                row[f"max{name}_ppc_trans_denorm"] = ppc_trans[f, T_ind]*tot_pass_cnt

            if frame_id == pass_forward_frame:
                f = np.ravel_multi_index((true_y_ind, true_x_ind), xx.shape)
                end_x, end_y = field_locs[f]

                row[f"TRUEPASS_x"] = end_x
                row[f"TRUEPASS_y"] = end_y
                row[f"TRUEPASS_T"] = T[true_T_ind]
                row[f"TRUEPASS_ppc_off"] = ppc_off[f, true_T_ind]
                row[f"TRUEPASS_ppc_def"] = ppc_def[f, true_T_ind]
                row[f"TRUEPASS_xyac"] = xyac[f, true_T_ind]
                row[f"TRUEPASS_xepa"] = xepa[f, true_T_ind]
                row[f"TRUEPASS_trans"] = trans_prob[f, true_T_ind]
                row[f"TRUEPASS_trans_denorm"] = trans_prob[f, true_T_ind]*tot_pass_cnt
                row[f"TRUEPASS_hist_trans"] = h_trans_prob[f, true_T_ind]
                row[f"TRUEPASS_hist_trans_denorm"] = h_trans_prob[f, true_T_ind]*tot_pass_cnt
                row[f"TRUEPASS_eppa"] = eppa[f, true_T_ind]
                row[f"TRUEPASS_eppa_denorm"] = eppa[f, true_T_ind]*tot_pass_cnt
                row[f"TRUEPASS_ppc_val"] = ppc_value[f, true_T_ind]
                row[f"TRUEPASS_ppc_trans"] = ppc_trans[f, true_T_ind]
                row[f"TRUEPASS_ppc_trans_denorm"] = ppc_trans[f, true_T_ind]*tot_pass_cnt

            passes_df = pd.DataFrame(row, index=[t])

        if viz_df:
            field_df = pd.DataFrame({
                'frameId': frame_id,
                'ball_end_x': field_locs[:, 0],
                'ball_end_y': field_locs[:, 1],
                'eppa': eppa.sum(axis=1),
                'trans': trans_prob.sum(axis=1),  # sum when pdf of T has been factored
                'ppc_off': ppc_off.mean(axis=1),  # otws mean for mean uniform T pdf assumption
                'ppc_def': ppc_def.mean(axis=1),
                'ppc_trans': (trans_prob*ppc_off).sum(axis=1),
                'xyac': xyac.mean(axis=1),
                'xepa': xepa.mean(axis=1),
            })
            field_df.loc[field_df.ball_end_x < ball_start[0]-10, :] = np.nan  # remove backward passes
        return field_df, passes_df, player_stats_df, proj_df

    field_dfs = pd.DataFrame()
    player_stats_df = pd.DataFrame()
    passes_df = pd.DataFrame()
    proj_df = pd.DataFrame()
    for fid in tqdm(range(ball_snap_frame+1, pass_forward_frame+1)):
        # for fid in tqdm(range(20, 22)):
        field_df, passes, player_stats, projs = frame_eppa(fid)
        field_dfs = field_dfs.append(field_df, ignore_index=True)
        passes_df = passes_df.append(passes, ignore_index=True)
        player_stats_df = player_stats_df.append(player_stats, ignore_index=True)
        proj_df = proj_df.append(projs, ignore_index=True)
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

# for (gid, pid) in tqdm(random.sample(plays, len(plays))):
for (gid, pid) in tqdm(plays):
    dir = out_dir_path.format(f'1/{gid}/{pid}')
    if os.path.exists(dir):
        print(f'EXISTS: {gid}, {pid}')
    else:
        try:
            play_eppa(gid, pid, viz_df=False, save_np=False, stats_df=True, viz_true_proj=True, save_all_dfs=True)
        except Exception as e:
            print(f"ERROR: {gid}, {pid}. e={e}")

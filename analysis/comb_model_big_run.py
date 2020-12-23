import code
import sys
import traceback
import random
import time
from joblib import Parallel, delayed
import multiprocessing as mp
from visualize import AnimatePlay
import treelite_runtime
import joblib
import xgboost as xgb
import warnings
from tqdm import tqdm
from IPython.display import HTML

import numpy as np
import pandas as pd
import os
from pathlib import Path

# file loading and prep
path_shared = '../data/{}'

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

out_dir_path = '../output/{}'  #for cloud runs
# out_dir_path = '/mnt/c/Users/uditr/OneDrive/Projects/BigDataBowl/output/{}'  #for local runs


params = lambda: None # create an empty object to add params
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

# historical trans model inputs/params
L_given_ts = np.load('in/L_given_t.npy')
T_given_Ls_df = pd.read_pickle('in/T_given_L.pkl')
# from L_given_t in historical notebook
x_min, x_max = -9, 70
y_min, y_max = -39, 40
t_min, t_max = 10, 63

#epa/xyac model loading
bst = joblib.load("./in/xyac_model.model")
cols_when_model_builds = bst.feature_names
xyac_predictor = treelite_runtime.Predictor('./in/xyacmymodel.so')
epa_model = joblib.load("./in/epa_model_rishav_no_time.model")
cols_when_model_builds_ep = epa_model.feature_names
epa_predictor = treelite_runtime.Predictor('./in/epa_no_time_mymodel.so')

def play_eppa(game_id, play_id, viz_df=True, save_np=False):
    play_df = track_df[(track_df.playId == play_id) & (track_df.gameId == game_id)].sort_values(by = 'frameId')
    
    ball_snap_frame = play_df.loc[(play_df.nflId == 0) & (play_df.event == 'ball_snap')].frameId.iloc[0]
    pass_forward_frame = play_df.loc[(play_df.nflId == 0) & (play_df.event == 'pass_forward')].frameId.iloc[0]
    play_df['frames_since_snap'] = play_df.frameId - ball_snap_frame
    
    # per play epa model
    def getEPAModel():
        epvals = np.array([7,-7,3,-3,2,-2,0])
        joined_df = pbp_joined[(pbp_joined.playId == play_id) & (pbp_joined.gameId == game_id)]
        epa_df = pd.DataFrame({'play_endpoint_x': x})

        test = {}
        for feat in epa_model.feature_names:
            test[feat] = [joined_df.iloc[0][feat]]
            epa_df[feat] = joined_df.iloc[0][feat]

        first_df = pd.DataFrame(test)

        dtest = treelite_runtime.Batch.from_npy2d(first_df[cols_when_model_builds_ep].values)
        ypred = epa_predictor.predict(dtest)
        ep = np.sum(ypred*epvals, axis = 1)
#         print(ep)
#         print("YAY")
        epa_df["before_play_ep"] = ep[0]

        epa_df['los'] = 110 - epa_df['yardline_100']
        epa_df['first_down_line'] = epa_df['los'] + epa_df["ydstogo"]
        epa_df['expected_end_up_line'] = epa_df['play_endpoint_x']

        epa_df['isFirstDown'] = 0
        epa_df['isFirstDown'] = np.where(epa_df['expected_end_up_line'] >= epa_df['first_down_line'], 1, 0)

        epa_df['yardline_100'] = np.round(110 - epa_df['expected_end_up_line'])
        epa_df['yardline_100'] = np.clip(epa_df['yardline_100'], 0 , 100)
        epa_df['ydstogo'] = epa_df['first_down_line'] - epa_df['expected_end_up_line']
        epa_df['ydstogo'] = np.where(epa_df['isFirstDown'] == 1, 10, epa_df['ydstogo'])
        old_down = joined_df.iloc[0]['down_x']
#         print(downthing)
        for d in range(1,6):
            epa_df['down%d' % d] = 1 if (d == old_down+1) else 0

        epa_df['down1'] = np.where(epa_df['isFirstDown'] == 1, 1, epa_df['down1'])
        epa_df['down2'] = np.where(epa_df['isFirstDown'] == 1, 0, epa_df['down2'])
        epa_df['down3'] = np.where(epa_df['isFirstDown'] == 1, 0, epa_df['down3'])
        epa_df['down4'] = np.where(epa_df['isFirstDown'] == 1, 0, epa_df['down4'])
        epa_df['down5'] = np.where(epa_df['isFirstDown'] == 1, 0, epa_df['down5']) # down 5 means turnover on downs

        # offense is other team inverted if turnover on downs
        epa_df['yardline_100'] = np.where(epa_df.down5 == 1, 100-epa_df['yardline_100'], epa_df['yardline_100'])
        epa_df['ydstogo'] = np.where(epa_df.down5 == 1, 10, epa_df['ydstogo'])
        epa_df['down1'] = np.where(epa_df.down5 == 1, 1, epa_df['down1'])

        ### UPDATE EPA VARIABLES ###

        dtest = treelite_runtime.Batch.from_npy2d(epa_df[cols_when_model_builds_ep].values)
        ypred = epa_predictor.predict(dtest)
        ep = np.sum(ypred*epvals, axis = 1)
        epa_df['xep'] = ep #ep after play
        ### SCORE SAFETIES
        epa_df['xep'] = np.where(epa_df['play_endpoint_x'] <= 10, -2, epa_df['xep'])
        ### SCORE Tds 
        epa_df['xep'] = np.where(epa_df['play_endpoint_x'] >= 110, 7, epa_df['xep'])

        epa_df['xepa'] = np.where(epa_df.down5 == 1, -epa_df['xep'] - epa_df['before_play_ep'], epa_df['xep'] - epa_df['before_play_ep']) #if turnover 

        only_vals = epa_df[["play_endpoint_x", "xep", "xepa"]] # THIS CONTAINS THE EPA VALUES BASED ON PLAY ENDPOINT
        return only_vals
    
    epa_df = getEPAModel()
    
    # input: tracking data for a single frame
    # output: frame_eppa (F, T)... (writes intermediate F,T to disk)
    def frame_eppa(frame_id):
        frame_df = play_df.loc[play_df.frameId==frame_id].reset_index()
        
        # per frame shared calcs
        ball_start = frame_df.loc[frame_df.position=='QB', ['x', 'y']].iloc[0].to_numpy(dtype=dt)
        t = frame_df.frames_since_snap.iloc[0]
        frame_df = frame_df.loc[(frame_df.nflId!=0) & (frame_df.position!='QB')] # remove ball and qb from df
        
        reach_vecs = (field_locs - ball_start).astype(dt)  # (F, 2)
        reach_dist = (np.linalg.norm(reach_vecs, axis=1)).astype(dt)  # (F,)

        # project motion by reaction time
        frame_df['v_x_r'] = frame_df.a_x*params.reax_t+frame_df.v_x
        frame_df['v_y_r'] = frame_df.a_y*params.reax_t+frame_df.v_y
        frame_df['v_r_mag'] = np.linalg.norm(np.array([frame_df.v_x_r, frame_df.v_y_r], dtype=dt), axis=0)
        frame_df['v_r_theta'] = np.arctan(frame_df.v_y_r/frame_df.v_x_r).fillna(0)
        frame_df['x_r'] = frame_df.x + frame_df.v_x*params.reax_t + 0.5*frame_df.a_x*params.reax_t**2
        frame_df['y_r'] = frame_df.y + frame_df.v_y*params.reax_t + 0.5*frame_df.a_y*params.reax_t**2

        player_teams = frame_df['team_pos'].to_numpy() # J,
        player_ids = frame_df['nflId'].to_numpy()
        reaction_player_locs = frame_df[['x_r', 'y_r']].to_numpy(dtype=dt) # (J, 2)
        reaction_player_vels = frame_df[['v_x_r', 'v_y_r']].to_numpy(dtype=dt) #(J,2)

        # intercept vector between each player and field location
        int_d_vec = field_locs[:, None, :] - reaction_player_locs #F, J, 2
        int_d_mag = np.linalg.norm(int_d_vec, axis=2) # F, J
        #projecting player velocity on d_vec to get initial speed along d_vec
        int_s0 = np.clip(np.sum(int_d_vec*reaction_player_vels, axis=2)/int_d_mag, -params.s_max, params.s_max) #F, J,

        # calculate time to int based on phys model
        t_lt_smax = (params.s_max-int_s0)/params.a_max  #F, J,
        d_lt_smax = t_lt_smax*((int_s0+params.s_max)/2) #F, J,
        d_at_smax = int_d_mag - d_lt_smax               #F, J,
        t_at_smax = d_at_smax/params.s_max              #F, J,
        t_tot = t_lt_smax+t_at_smax+params.reax_t       #F, J,

        # int success if T-t_tot = dT <  0. Put through sigmoid to add temporal uncertainty around 
        int_dT = T[None,:,None] - t_tot[:,None,:]         #F, T, J
        p_int = (1/(1. + np.exp( -np.pi/np.sqrt(3.0)/params.tti_sigma * int_dT, dtype=dt) )) #F, T, J

        # input: qb_loc (2,), t=frames_after_snap (int)
        # output: (P(L,T)|t) int probability of each pass (F, T)
        def hist_trans_prob():
            """ P(L|t) """
            ball_start_ind = np.rint(ball_start).astype(int)
#             mask for zeroing out parts of the field that are too far to be thrown to per the L_given_t model
            L_t_mask = np.zeros_like(xx, dtype=dt)  # (Y, X)
            L_t_mask[max(ball_start_ind[1]+y_min,0):min(ball_start_ind[1]+y_max,len(y)-1),\
                     max(ball_start_ind[0]+x_min,0):min(ball_start_ind[0]+x_max,len(x)-1)] = 1.
            L_t_mask = L_t_mask.flatten()  # (F,)
#             # we clip reach vecs to be used to index into L_given_t.
#             # eg if qb is far right, then the left field will be clipped to y=-39 and later zeroed out
#             reach_vecs_int = np.rint(reach_vecs).astype(int)
            
#             clipped_reach_vecs = np.stack((np.clip(reach_vecs_int[:,0], x_min, x_max),
#                                           np.clip(-reach_vecs_int[:,1], y_min, y_max)))  # (2, F)
#             t_i = max(t-t_min, 0)
#             L_given_t = L_given_ts[t_i, clipped_reach_vecs[1]-y_min, clipped_reach_vecs[0]-x_min] * L_t_mask  # (F,) ; index with y and then x
            L_given_t = L_t_mask #changed L_given_t to uniform after discussion
            # L_given_t /= L_given_t.sum()  # renormalize since part of L|t may have been off field

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

            L_T_given_t = L_given_t[:,None] * T_given_L #(F, T)
            L_T_given_t /= L_T_given_t.sum() # normalize all passes after some have been chopped off
            return L_T_given_t

        def comp_prob():
            # use p_int as memoized values for integration

            # trajectory integration
            g = 10.72468 #y/s/s
            dx = reach_vecs[:, 0] #F
            dy = reach_vecs[:, 1] #F
            vx = dx[:, None]/T[None, :]   #F, T
            vy = dy[:, None]/T[None, :]   #F, T
            vz_0 = (T*g)/2                #T

            # note that idx (i, j, k) into below arrays is invalid when j < k
            traj_ts = np.tile(T, (len(field_locs), len(T), 1)) #(F, T, T)
            traj_locs_x_idx = np.rint(np.clip((ball_start[0]+vx[:, :, None]*T), 0, len(x)-1)).astype(int) # F, T, T
            traj_locs_y_idx = np.rint(np.clip((ball_start[1]+vy[:, :, None]*T), 0, len(y)-1)).astype(int) # F, T, T
            traj_locs_z = 2.0+vz_0[None, :, None]*traj_ts-0.5*g*traj_ts*traj_ts #F, T, T
            path_idxs = np.ravel_multi_index(np.stack((traj_locs_y_idx, traj_locs_x_idx)).reshape(2, -1), xx.shape)  # (F*T*T,)
            traj_t_idxs = np.rint(10*traj_ts - 1).flatten().astype(int)  # (F, T, T)
            p_int_traj = p_int[path_idxs, traj_t_idxs] # F*T*T, J
            p_int_traj = p_int_traj.reshape((*traj_locs_x_idx.shape, len(reaction_player_locs)))  # F, T, T, J

            # account for ball height on traj and normalize each locations int probability
            lambda_z = np.where((traj_locs_z<params.z_max)&(traj_locs_z>params.z_min), 1, 0) #F, T, T # maybe change this to a normal distribution 
            p_int_traj = p_int_traj * lambda_z[:, :, :, None]
            norm_factor = np.maximum(1., p_int_traj.sum(axis=-1))  #F, T, T
            p_int_traj_norm = (p_int_traj/norm_factor[..., None])  #F, T, T, J

            # independent int probs at each point on trajectory
            all_p_int_traj = np.sum(p_int_traj_norm, axis=-1)  # F, T, T
            off_p_int_traj = np.sum(p_int_traj_norm, axis=-1, where=(player_teams=='OFF'))
            def_p_int_traj = np.sum(p_int_traj_norm, axis=-1, where=(player_teams=='DEF'))
            ind_p_int_traj = p_int_traj_norm #use for analyzing specific players

            # calc decaying residual probs after you take away p_int on earlier times in the traj 
            compl_all_p_int_traj = 1-all_p_int_traj  # F, T, T
            remaining_compl_p_int_traj = np.cumprod(compl_all_p_int_traj, axis=-1)  # F, T, T
            # maximum 0 because if it goes negative the pass has been caught by then and theres no residual probability
            shift_compl_cumsum = np.roll(remaining_compl_p_int_traj, 1, axis=-1)  # F, T, T
            shift_compl_cumsum[:, :, 0] = 1

            # multiply residual prob by p_int at that location
            all_completion_prob_dt = shift_compl_cumsum * all_p_int_traj  # F, T, T
            off_completion_prob_dt = shift_compl_cumsum * off_p_int_traj  # F, T, T
            def_completion_prob_dt = shift_compl_cumsum * def_p_int_traj  # F, T, T
            ind_completion_prob_dt = shift_compl_cumsum[:, :, :, None] * ind_p_int_traj  # F, T, T, J

            # now accumulate values over total traj for each team and take at T=t
            all_completion_prob = np.cumsum(all_completion_prob_dt, axis=-1)  # F, T, T
            off_completion_prob = np.cumsum(off_completion_prob_dt, axis=-1)  # F, T, T
            def_completion_prob = np.cumsum(def_completion_prob_dt, axis=-1)  # F, T, T
            ind_completion_prob = np.cumsum(ind_completion_prob_dt, axis=-2)  # F, T, T, J

                #     #### Toy example
        #         all_p_int_traj = [0, 0, 0.1, 0.2, 0.8, 0.8]
        #         c_all_p_int_traj=[1, 1, 0.9, 0.8, 0.2, 0.2]
        #         rem_compl_p_int_traj = [1, 1, 0.9, 0.72, 0.144, 0.0288]
        #         0.1 + 0.9*0.2 + 0.72 * 0.8 + 0.144*0.8 = 0.9712
        #         adjust_compl_prob =        [0, 0, 0.1, 0.28, 0.84, 0.84]


            # this einsum takes the diagonal values over the last two axes where T = t
            # this takes care of the t > T issue.
            all_p_int_pass = np.einsum('ijj->ij', all_completion_prob)  # F, T
            off_p_int_pass = np.einsum('ijj->ij', off_completion_prob)  # F, T
            def_p_int_pass = np.einsum('ijj->ij', def_completion_prob)  # F, T
            ind_p_int_pass = np.einsum('ijjk->ijk', ind_completion_prob)  # F, T, J
            no_p_int_pass = 1-all_p_int_pass #F, T

            # assert np.allclose(all_p_int_pass, off_p_int_pass + def_p_int_pass, atol=0.01)
            # assert np.allclose(all_p_int_pass, ind_p_int_pass.sum(-1), atol=0.01)
            return off_p_int_pass, def_p_int_pass, ind_p_int_pass

            # below gets cutoff for combined model
            # this is only for viz (in reality want to keep F, T above and mult by value/transition prob)
    #         field_p_int_all = all_p_int_pass.mean(axis=1)  # F,
    #         field_p_int_off = off_p_int_pass.mean(axis=1)  # F, 
    #         field_p_int_def = def_p_int_pass.mean(axis=1)  # F,
    #         field_p_no_int = 1-field_p_int_all
    #         assert np.allclose(field_p_int_all, field_p_int_off + field_p_int_def, atol=0.01)
    #         assert np.all(field_p_int_all <= 1.01) and np.all(field_p_int_all >= -0.01)
    #         assert np.all(field_p_int_off <= 1.01) and np.all(field_p_int_off >= -0.01)
    #         assert np.all(field_p_int_def <= 1.01) and np.all(field_p_int_def >= -0.01)

    #         field_df = pd.DataFrame({
    #             'ball_start_x': ball_start[0],
    #             'ball_start_y': ball_start[1], 
    #             'ball_end_x': field_locs[:,0],
    #             'ball_end_y': field_locs[:,1],
    #             'p_mass_1': (((field_p_int_off-field_p_int_def)+1.)/2.).round(3),
    #             'p_mass_2': field_p_no_int.round(3),
    #             # 'p_mass_players': p_int_norm,
    #         })

    #         return field_df

        def xyac():
            receivers_df = frame_df.loc[frame_df.team_pos == 'DEF',['x_r', 'y_r', 'v_x_r', 'v_y_r', 'v_r_theta', 'v_r_mag', 'los', 'a_x', 'a_y']]
#             dist_from_ball_np = np.linalg.norm((receivers_df.x - ball_start[0],
#                                                              receivers_df.y - ball_start[1]), axis=0)
            # find the spot the qb would aim at, leading the receiver in their current dir by the ball time
    #         rec_x_np = receivers_df.x.to_numpy()[:,None]
    #         rec_y_np = receivers_df.y.to_numpy()[:,None]
    #         rec_v_x_np = receivers_df.v_x.to_numpy()[:,None]
    #         rec_v_y_np = receivers_df.v_y.to_numpy()[:,None]
    #         rec_a_x_np = receivers_df.a_x.to_numpy()[:,None]
    #         rec_a_y_np = receivers_df.a_y.to_numpy()[:,None]
    #         rec_v_theta_np = receivers_df.v_theta.to_numpy()[:,None]
            rec_v_x_r = receivers_df.v_x_r.to_numpy()[:,None]
            rec_v_y_r = receivers_df.v_y_r.to_numpy()[:,None]
            rec_v_r_mag = receivers_df.v_r_mag.to_numpy()[:,None]
            rec_v_r_theta = receivers_df.v_r_theta.to_numpy()[:,None]
            x_r = receivers_df.x_r.to_numpy()[:,None] #(J, 1)
            y_r = receivers_df.y_r.to_numpy()[:,None] #(J, 1)

            #target_x = rec_x_np+rec_v_x_np*t+0.5*rec_a_x_np*t**2  # (R, T)
            #target_y = rec_y_np+rec_v_y_np*t+0.5*rec_a_y_np*t**2  # (R, T)

    #         x_r = rec_x_np + rec_v_x_np*params.reax_t + 0.5*rec_a_x_np*params.reax_t**2 #(J, 1)
    #         y_r = rec_y_np + rec_v_y_np*params.reax_t + 0.5*rec_a_y_np*params.reax_t**2 #(J, 1)

            reaction_player_locs = np.hstack((x_r, y_r)) # (J,2)
            reaction_player_vels = np.hstack((rec_v_x_r, rec_v_y_r)) # (J,2)

            int_d_vec = field_locs[:, None, :] - reaction_player_locs #(F, J, 2)
            int_d_mag = np.linalg.norm(int_d_vec, axis=2) # F, J
            int_theta = np.arctan(int_d_vec[:,:,1]/int_d_vec[:,:,0]) #this could be a problem


            int_s0 = np.clip(np.sum(int_d_vec*reaction_player_vels, axis=2)/int_d_mag, -params.s_max, params.s_max) #F, J,  #taking norm between vectors int_d and player velocity

            t_lt_smax = (params.s_max-int_s0)/params.a_max  #F, J,
            d_lt_smax = t_lt_smax*((int_s0+params.s_max)/2) #F, J,
            d_at_smax = int_d_mag - d_lt_smax               #F, J,
            t_at_smax = d_at_smax/params.s_max              #F, J,
            t_tot = t_lt_smax+t_at_smax+params.reax_t       #F, J,


            t_without_react = t_tot - params.reax_t 
            tempT = np.clip(T - params.reax_t, 0, 4)

            cap = np.broadcast_to(t_without_react[:,:,None],(*t_lt_smax.shape, len(T))) 
            a1 =  np.broadcast_to(t_lt_smax[:,:,None],(*t_lt_smax.shape, len(T))) # F, J, T THIS IS TIME SPENT LT SMAX
            a2 =  np.broadcast_to(tempT[None,None,:],(*t_lt_smax.shape, len(T))) # F, J, T
            newT = np.where(a2 > cap, cap, a2)
            a3 =  np.broadcast_to(newT,(*t_lt_smax.shape, len(T))) # F, J, T

            time_lt_smax = np.where(a1 > a3, a3, a1) # F, J, T  THIS IS TIME LESS THAN MAX
            time_at_smax = a3 - time_lt_smax # F, J, T THIS IS TIME MORE at max



            d = time_at_smax * params.s_max + int_s0[:,:,None]*time_lt_smax + 0.5*params.a_max*np.square(time_lt_smax) # F, J, T
            # d should be at most the magnitude

            #print(d)
            #print(int_s0)
            ## d = Time at max speed * max speed + int_s0* time at lt_smax + 1/2 params.a_max (time at lt_smax * time at lt_smax) 
            #d = 
            v_proj = int_s0[:,:,None] + params.a_max*time_lt_smax # F, J, T
            v_proj = np.where(v_proj > params.s_max, params.s_max, v_proj) # F, J, T
            #v_proj = np.abs(v_proj)
            x_proj = x_r + d* np.cos(int_theta[:,:,None]) # F, J, T
            y_proj = y_r + d* np.sin(int_theta[:,:,None]) # F, J, T

            x_proj_relv = x_proj -  field_locs[:,None,None,0] # F, J, T #GET RELATIVE COORDS
            y_proj_relv = y_proj -  field_locs[:,None,None,1] # F, J, T #GET RELATIVE COORDS


            projected_locations = np.stack((x_proj, y_proj), axis =3)  # F, J, T, 2

            distances_to_ball = projected_locations - field_locs[:,None,None,:] # F, J, T, 2
            distance_mags = np.linalg.norm(distances_to_ball, axis = 3) # F, J, T

            sorted_indices = np.argsort(distance_mags, axis = 1) # F, J, T

            distance_mags = np.take_along_axis(distance_mags,sorted_indices, axis = 1)
            x_proj_sorted = np.take_along_axis(x_proj_relv,sorted_indices, axis = 1) # F, J, T
            y_proj_sorted = np.take_along_axis(y_proj_relv,sorted_indices, axis = 1) # F, J, T
            v_proj_sorted = np.take_along_axis(v_proj,sorted_indices, axis = 1) # F, J, T

            just_top_5_distances = distance_mags[:,0:5,:].transpose((0,2,1)) #F, T, 5
            just_top_5_x_proj = x_proj_sorted[:,0:5,:].transpose((0,2,1)) #F, T, 5
            just_top_5_y_proj = y_proj_sorted[:,0:5,:].transpose((0,2,1))  #F, T, 5
            just_top_5_v_proj = v_proj_sorted[:,0:5,:].transpose((0,2,1))  #F, T, 5


            just_top_5_distances = np.reshape(just_top_5_distances, (just_top_5_distances.shape[0]*just_top_5_distances.shape[1],just_top_5_distances.shape[2]))
            just_top_5_x_proj  = np.reshape(just_top_5_x_proj, just_top_5_distances.shape)
            just_top_5_y_proj  = np.reshape(just_top_5_y_proj, just_top_5_distances.shape)
            just_top_5_v_proj  = np.reshape(just_top_5_v_proj, just_top_5_distances.shape)


            endpoints = np.repeat(field_locs, repeats = len(T), axis = 0) # FxT, 2
#             assert((field_locs[:, None, :]+np.zeros_like(T[None, None, :])).reshape(end))
            times = np.repeat(T[None, :], repeats = len(field_locs), axis = 0)
            times_shaped = times.reshape((times.shape[0]*times.shape[1]))# FxT, 1
            value_array = np.array([-2.5,2.5,7.5,12.5,17.5, 22.5, 27.5, 30])


            field_df = pd.DataFrame({
                'pass_endpoint_x': endpoints[:,0],
                'pass_endpoint_y': endpoints[:,1],
                'frame_thrown' : t,
                'time_of_flight' : times_shaped,
                '1-closest-defender-distance' : just_top_5_distances[:,0],
                '2-closest-defender-distance' : just_top_5_distances[:,1],
                '3-closest-defender-distance' : just_top_5_distances[:,2],
                '4-closest-defender-distance' : just_top_5_distances[:,3],
                '5-closest-defender-distance' : just_top_5_distances[:,4],
                '1-closest-defender-x' : just_top_5_x_proj[:,0],
                '2-closest-defender-x' : just_top_5_x_proj[:,1],
                '3-closest-defender-x' : just_top_5_x_proj[:,2],
                '4-closest-defender-x' : just_top_5_x_proj[:,3],
                '5-closest-defender-x' : just_top_5_x_proj[:,4],
                '1-closest-defender-y' : just_top_5_y_proj[:,0], 
                '2-closest-defender-y': just_top_5_y_proj[:,1], 
                '3-closest-defender-y': just_top_5_y_proj[:,2], 
                '4-closest-defender-y': just_top_5_y_proj[:,3], 
                '5-closest-defender-y': just_top_5_y_proj[:,4], 
                '1-closest-defender-speed' : just_top_5_v_proj[:,0],
                '2-closest-defender-speed': just_top_5_v_proj[:,1],
                '3-closest-defender-speed': just_top_5_v_proj[:,2],
                '4-closest-defender-speed': just_top_5_v_proj[:,3],
                '5-closest-defender-speed': just_top_5_v_proj[:,4], 
                "y" : endpoints[:,1]

             })
            ### CALCULTE XYAC

            dtest = treelite_runtime.Batch.from_npy2d(field_df[cols_when_model_builds].values)
            ypred = xyac_predictor.predict(dtest)
            y_vals = np.sum(ypred*value_array, axis = 1)
            field_df['xyac'] = y_vals
            field_df['play_endpoint_x'] = np.round(field_df['xyac'] + field_df['pass_endpoint_x'])
            field_df['play_endpoint_x'] = field_df['play_endpoint_x']+.5
            field_df['play_endpoint_x'] = np.clip(field_df['play_endpoint_x'], 0.5, 119.5)

            return field_df[['pass_endpoint_x', 'pass_endpoint_y', 'time_of_flight', 'xyac', 'play_endpoint_x']]

        
        nonlocal epa_df
        
        ppc_off, ppc_def, ppc_ind = comp_prob() #(F, T), (F, T), (F, T, J)
        ind_info = np.stack((player_ids, player_teams), axis=1)
        h_trans_prob = hist_trans_prob()
        
        epa_xyac_df = xyac().merge(epa_df, how='left', on='play_endpoint_x')
        # xyac = epa_xyac_df.xyac.to_numpy().reshape(ppc_off.shape)
        # end_x = epa_xyac_df.play_endpoint_x.to_numpy().reshape(ppc_off.shape)
        xepa = epa_xyac_df.xepa.to_numpy().reshape(ppc_off.shape)
        # assert(h_trans_prob.shape == ppc_off.shape)

        ppc = ppc_off
        trans_prob = h_trans_prob * np.power(ppc, params.alpha) #F, T
        trans_prob /= trans_prob.sum()


        eppa = ppc*trans_prob*xepa
        eppa_ind = ppc_ind*trans_prob[:, :, None]*xepa[:, :, None]
        
        if save_np:
            week = games_df.loc[games_df.gameId==game_id].week.to_numpy()[0].item()
            dir = out_dir_path.format(f'{week}/{game_id}/{play_id}')
            Path(dir).mkdir(parents=True, exist_ok=True)
            # np.savez_compressed(f'{dir}/{frame_id}', players_ind_info=ind_info,
            #                     ppc_ind=ppc_ind, h_trans=h_trans_prob, xepa=xepa, eppa_ind=eppa_ind)
            np.savez_compressed(f'{dir}/{frame_id}', ind_info=ind_info, eppa_ind=eppa_ind)
        if viz_df:
            ppc_no = 1-ppc_off-ppc_def
            xyac = epa_xyac_df.xyac.to_numpy().reshape(ppc_off.shape)
            end_x = epa_xyac_df.play_endpoint_x.to_numpy().reshape(ppc_off.shape)

            field_df = pd.DataFrame({
                'frameId': frame_id,
                'ball_end_x': field_locs[:,0],
                'ball_end_y': field_locs[:,1],
                'eppa': eppa.sum(axis=1),
                'hist_trans': h_trans_prob.sum(axis=1),
                'trans': trans_prob.sum(axis=1), #sum when pdf of T has been factored
                'ppcf_off': ppc_off.mean(axis=1), # otws mean for mean uniform T pdf assumption
                'ppcf_def': ppc_def.mean(axis=1),
                'ppcf_no': ppc_no.mean(axis=1),
                'ht_ppc_off': (h_trans_prob*ppc_off).sum(axis=1),
                'ht_ppc_def': (h_trans_prob*ppc_def).sum(axis=1),
                't_ppc_off': (trans_prob*ppc_off).sum(axis=1),
                't_ppc_def': (trans_prob*ppc_def).sum(axis=1),
                'xyac': xyac.mean(axis=1),
                'xepa': xepa.mean(axis=1),
            })
            field_df.loc[field_df.ball_end_x<ball_start[0]-10, :] = np.nan # remove backward passes
            return field_df
        
        return pd.DataFrame() #empty df if no viz
    
    field_dfs = pd.DataFrame()
    for fid in tqdm(range(ball_snap_frame, pass_forward_frame+1)):
        field_dfs = field_dfs.append(frame_eppa(fid))
    return play_df, field_dfs


# def parr_eppa(ids):
#      gid, pid = ids[0], ids[1]
#      print(gid, pid)
#      try:
#          play_eppa(gid, pid, viz_df=False, save_np=True)
#      except:
#          pass


# inputs = plays[:2]

# pool = mp.Pool(processes=1)
# pool.map(parr_eppa, inputs)

plays = sorted(list(set(map(lambda x: (x[0].item(), x[1].item()), track_df.groupby(
    ['gameId', 'playId'], as_index=False).first()[['gameId', 'playId']].to_numpy()))))

for (gid, pid) in tqdm(plays):
    print(gid, pid)
    try:
        play_eppa(gid, pid, viz_df=False, save_np=True)
    except:
        pass

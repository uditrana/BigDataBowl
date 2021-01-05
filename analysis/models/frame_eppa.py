import pandas as pd
import numpy as np
import torch
import xgboost as xgb
#import treelite
#import treelite_runtime
from .consts import *
from .params import params
import joblib


# file loading and prep
path_shared = '../data/{}'

dt = np.float64
dt_torch = torch.float64
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def torchify(np_arr):
    return torch.from_numpy(np_arr).to(device)

T_torch = torchify(T)
xx_torch = torchify(xx)
yy_torch = torchify(yy)
field_locs_torch = torchify(field_locs)

games_df = pd.read_csv(path_shared.format('games.csv'))
plays_df = pd.read_csv(path_shared.format('plays.csv'))
players_df = pd.read_csv(path_shared.format('players.csv'))
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

# historical trans model inputs/params
L_given_ts = np.load('models/in/L_given_t.npy')
T_given_Ls = pd.read_csv('models/in/T_given_L.csv')['p'].values.reshape(60, len(T))  # (61, T)
# from L_given_t in historical notebook
x_min, x_max = -9, 70
y_min, y_max = -39, 40
t_min, t_max = 10, 63

# epa/xyac model loading
bst = joblib.load("models/in/xyac_model.model")
bst.set_param({'predictor': 'gpu_predictor'})
scores = bst.get_score(importance_type='gain')
cols_when_model_builds = bst.feature_names
xyac_predictor = bst  # treelite_runtime.Predictor('models/in/xyacmymodel.dylib')
epa_model = joblib.load("models/in/epa_model_rishav_no_time.model")
epa_model.set_param({'predictor': 'gpu_predictor'})
scores = epa_model.get_score(importance_type='gain')
cols_when_model_builds_ep = epa_model.feature_names
epa_predictor = epa_model  # treelite_runtime.Predictor('models/in/epa_no_time_mymodel.dylib')


# per play epa model
def getEPAModel(game_id, play_id):
    epvals = np.array([7, -7, 3, -3, 2, -2, 0])
    joined_df = pbp_joined[(pbp_joined.playId == play_id) & (pbp_joined.gameId == game_id)]
    epa_df = pd.DataFrame({'play_endpoint_x': x})

    test = {}
    for feat in epa_model.feature_names:
        test[feat] = [joined_df.iloc[0][feat]]
        epa_df[feat] = joined_df.iloc[0][feat]

    first_df = pd.DataFrame(test)

    dtest = xgb.DMatrix(first_df[cols_when_model_builds_ep])  # treelite_runtime.Batch.from_npy2d(first_df[cols_when_model_builds_ep].values)
    ypred = epa_predictor.predict(dtest)
    ep = np.sum(ypred*epvals, axis=1)
    epa_df["before_play_ep"] = ep[0]
    ###### EP FOR A INCOMPLETE #######
    epa_df_incomp = epa_df.copy(deep=True)
    old_down = joined_df.iloc[0]['down_x']
    epa_df_incomp['isFirstDown'] = 0
    for d in range(1, 6):
        epa_df_incomp['down%d' % d] = 1 if (d == old_down+1) else 0
    # offense is other team inverted if turnover on downs
    epa_df_incomp['yardline_100'] = np.where(epa_df_incomp.down5 == 1, 100-epa_df_incomp['yardline_100'], epa_df_incomp['yardline_100'])
    epa_df_incomp['ydstogo'] = np.where(epa_df_incomp.down5 == 1, 10, epa_df_incomp['ydstogo'])
    epa_df_incomp['down1'] = np.where(epa_df_incomp.down5 == 1, 1, epa_df_incomp['down1'])
    # treelite_runtime.Batch.from_npy2d(epa_df_incomp[cols_when_model_builds_ep].values)
    dtest = xgb.DMatrix(epa_df_incomp[cols_when_model_builds_ep])
    ypred = epa_predictor.predict(dtest)
    ep = np.sum(ypred*epvals, axis=1)
    epa_df['xep_inc'] = ep
    epa_df['xepa_inc'] = np.where(
        epa_df_incomp.down5 == 1, -epa_df['xep_inc'] - epa_df_incomp['before_play_ep'],
        epa_df['xep_inc'] - epa_df_incomp['before_play_ep'])

    ###### EP FOR A CATCH #######

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

    dtest = xgb.DMatrix(epa_df[cols_when_model_builds_ep])  # treelite_runtime.Batch.from_npy2d(epa_df[cols_when_model_builds_ep].values)
    ypred = epa_predictor.predict(dtest)
    ep = np.sum(ypred*epvals, axis=1)
    epa_df['xep'] = ep  # ep after play
    # SCORE SAFETIES
    epa_df['xep'] = np.where(epa_df['play_endpoint_x'] <= 10, -2, epa_df['xep'])
    # SCORE Tds
    epa_df['xep'] = np.where(epa_df['play_endpoint_x'] >= 110, 7, epa_df['xep'])

    epa_df['xepa'] = np.where(epa_df.down5 == 1, -epa_df['xep'] - epa_df['before_play_ep'],
                              epa_df['xep'] - epa_df['before_play_ep'])  # if turnover

    only_vals = epa_df[["play_endpoint_x", "xep", "xepa", "xep_inc", "xepa_inc"]].rename(
        columns={'xep': 'xep_comp', 'xepa': 'xepa_comp'})  # THIS CONTAINS THE EPA VALUES BASED ON PLAY ENDPOINT
    return only_vals


def frame_eppa(play_df, frame_id):
    play_id = play_df.playId.iloc[0]
    game_id = play_df.gameId.iloc[0]
    frame_df = play_df.loc[play_df.frameId == frame_id].reset_index()
    ball_start = frame_df.loc[frame_df.position == 'QB', ['x', 'y']].iloc[0].to_numpy(dtype=dt)
    ball_start_torch = torchify(ball_start)
    t = frame_df.frames_since_snap.iloc[0]
    frame_df = frame_df.loc[(frame_df.nflId != 0) & (frame_df.position != 'QB')]  # remove ball and qb from df

    reach_vecs = (field_locs - ball_start).astype(dt)  # (F, 2)
    reach_vecs_torch = torchify(reach_vecs)

    frame_df = frame_df.drop_duplicates(subset='nflId').sort_values('nflId').reset_index()

    player_teams = frame_df['team_pos'].to_numpy()  # J,
    player_off = player_teams == 'OFF'
    player_def = np.logical_not(player_off)
    player_team_names = frame_df['teamAbbr'].to_numpy()  # J,
    player_ids = frame_df['nflId'].to_numpy()
    player_names = frame_df['displayName'].to_numpy()
    reaction_player_locs = frame_df[['x_opt', 'y_opt']].to_numpy(dtype=dt)  # (J, 2)
    reaction_player_vels = frame_df[['v_x_opt', 'v_y_opt']].to_numpy(dtype=dt)  # (J,2)

    player_off_torch = torchify(player_off)  # J,
    player_def_torch = torchify(player_def)  # J,

    # intercept vector between each player and field location
    int_d_vec = field_locs[:, None, :] - reaction_player_locs  # F, J, 2
    int_d_mag = np.linalg.norm(int_d_vec, axis=2) + 1e-3  # F, J
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
    p_int_adj = p_int.copy()
    p_int_def = np.power(1-np.prod((1-p_int_adj[:, :, player_def]), axis=-1), params.def_beta)  # F, T,
    p_int_adj[:, :, player_off] = p_int_adj[:, :, player_off]*(1-p_int_def[..., None])  # F, T, J
    p_int_off = 1-np.prod((1-p_int_adj[:, :, player_off]), axis=-1)  # F, T
    p_int_adj_torch = torchify(p_int_adj)

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
    # output: (P(L,T)|t) int probability of each pass (F, T)
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

        off_p_int_pass = 1-torch.prod(1-ind_p_int_pass * player_off_torch[None, None], dim=-1)  # F, T
        def_p_int_pass = 1-torch.prod(1-ind_p_int_pass * player_def_torch[None, None], dim=-1)  # F, T

        # assert np.allclose(all_p_int_pass, off_p_int_pass + def_p_int_pass, atol=0.01)
        # assert np.allclose(all_p_int_pass, ind_p_int_pass.sum(-1), atol=0.01)
        return off_p_int_pass.detach().cpu().numpy(),\
            def_p_int_pass.detach().cpu().numpy(),\
            ind_p_int_pass.detach().cpu().numpy()

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

        dtest = xgb.DMatrix(field_df[cols_when_model_builds])  # treelite_runtime.Batch.from_npy2d(field_df[cols_when_model_builds].values)
        ypred = xyac_predictor.predict(dtest)
        y_vals = np.sum(ypred*value_array, axis=1)
        field_df['xyac'] = y_vals
        field_df['play_endpoint_x'] = np.round(field_df['xyac'] + field_df['pass_endpoint_x'])
        field_df['play_endpoint_x'] = field_df['play_endpoint_x']+.5
        field_df['play_endpoint_x'] = np.clip(field_df['play_endpoint_x'], 0.5, 119.5)

        return field_df[['pass_endpoint_x', 'pass_endpoint_y', 'time_of_flight', 'xyac', 'play_endpoint_x']]

    epa_df = getEPAModel(game_id, play_id)

    ppc_off, ppc_def, ppc_ind = get_ppc()  # (F, T), (F, T), (F, T, J)

    # value model
    epa_xyac_df = get_xyac().merge(epa_df, how='left', on='play_endpoint_x')
    xyac = epa_xyac_df.xyac.to_numpy().reshape(ppc_off.shape)
    # end_x = epa_xyac_df.play_endpoint_x.to_numpy().reshape(ppc_off.shape)
    xepa_comp = epa_xyac_df.xepa_comp.to_numpy().reshape(ppc_off.shape)  # F, T
    xepa_inc = epa_xyac_df.xepa_inc.iloc[0]
    xepa_diff = xepa_comp-xepa_inc
    # assert(h_trans_prob.shape == ppc_off.shape)
    # breakpoint()

    # transition model
    h_trans_prob = get_hist_trans_prob()   # (F, T)
    trans = h_trans_prob * np.power(ppc_off, params.alpha)  # F, T
    trans /= trans.sum()

    # output metrics (eppa1 uses ppc_off as catch prob, eppa2 uses catch_prob)
    ind_eppa1_wo_value = ppc_ind * trans[..., None]  # F, T, J
    ind_eppa1 = ind_eppa1_wo_value * xepa_diff[..., None]  # F, T, J
    # ind_eppa2_wo_value = catch_prob_ind * trans[..., None]  # F, T, J
    # ind_eppa2 = ind_eppa2_wo_value * xepa_diff[..., None]  # F, T, J
    eppa1_pass_val = (ppc_off*xepa_comp)+(1-ppc_off)*xepa_inc
    eppa1 = eppa1_pass_val*trans  # F, T

    return eppa1

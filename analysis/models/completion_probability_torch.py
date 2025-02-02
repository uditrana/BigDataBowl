import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import os
#import dask.dataframe as dd
import torch
from enum import Enum
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from tqdm import tqdm
import argparse
import time
from utils import get_repo_dir


class TuningParam(Enum):
    sigma = 1
    lamb = 2
    alpha = 3
    av = 4

class PlaysDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, wk=1, all_weeks=False, event_filter=None, tuning=None):
        # time code TODO(adit98) remove this later
        start_time = time.time()

        self.tuning = tuning

        if all_weeks:
            all_data = []
            for week in range(5, 10):
                all_data.append(pd.read_csv(os.path.join(data_dir, 'week%d_norm.csv' % week)))

            tracking_df = pd.concat(all_data)

        else:
            # load csvs
            tracking_df = pd.read_csv(os.path.join(data_dir, 'week%s_norm.csv' % wk))

        print('loaded files', time.time() - start_time)

        # generate unique id from game, play, frame ids
        tracking_df['uniqueId'] = tracking_df[['gameId', 'playId', 'frameId']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)

        print('generated unique id', time.time() - start_time)

        # remove frames with more than 17 players' tracking data + QB + ball (19)
        tracking_df = tracking_df.loc[tracking_df.position != 'QB']
        #tracking_df = tracking_df.groupby(['uniqueId']).filter(lambda x: len(x.nflId.unique()) <= 19)

        # calculate pass outcome in tracking_df
        tracking_df['pass_outcome'] = tracking_df['event'].copy().replace({'pass_forward': 0, 'pass_arrived': 0,
            'pass_outcome_incomplete': 0, 'pass_outcome_interception': 0, 'pass_outcome_caught': 1, 'pass_outcome_touchdown': 1})
        tracking_df.loc[(tracking_df.pass_outcome != 0) & (tracking_df.pass_outcome != 1), 'pass_outcome'] = 0 # everything else is 0
        tracking_df['pass_outcome'] = tracking_df.groupby(['uniqueId']).pass_outcome.transform('max')

        print('calculated pass outcome', time.time() - start_time)

        # get valid frames for tuning from tracking df (consider every pass, labels are 1 if there is a player close by)
        pass_forward_plays = tracking_df.loc[tracking_df['event'] == 'pass_forward', 'uniqueId'].copy().drop_duplicates()
        pass_arrived_plays = tracking_df.loc[tracking_df['event'] == 'pass_arrived', 'uniqueId'].copy().drop_duplicates()
        tracking_df = tracking_df.loc[(tracking_df.uniqueId.isin(pass_forward_plays)) | (tracking_df.uniqueId.isin(pass_arrived_plays))]

        # get forward_frameId, arrived_frameId which contains first frame of pass_forward, pass_arrived for each play
        tracking_df['first_frameId'] = tracking_df.groupby(['gameId', 'playId', 'event']).frameId.transform('min')

        print('calculated first frame of each event', time.time() - start_time)

        # calculate ball ending position
        ball_end = tracking_df.loc[(tracking_df.nflId == 0) & (tracking_df.event == 'pass_arrived'), ['gameId', 'playId', 'x', 'y']].copy()
        ball_end = ball_end.rename(columns={'x': 'ball_end_x', 'y': 'ball_end_y'}).drop_duplicates()

        # calculate ball position at throw
        ball_start = tracking_df.loc[(tracking_df.nflId == 0) & (tracking_df.event == 'pass_forward'), ['gameId', 'playId', 'x', 'y']].copy()
        ball_start = ball_start.rename(columns={'x': 'ball_start_x', 'y': 'ball_start_y'}).drop_duplicates()

        # merge into single df (could make this faster with a simple concatenate)
        ball_start_end = ball_end.merge(ball_start, on=['gameId', 'playId'])

        # remove plays where ball is thrown out of bounds
        ball_start_end = ball_start_end.loc[(ball_start_end.ball_end_x <= 119.5) & (ball_start_end.ball_end_x >= 0.5) & \
                (ball_start_end.ball_end_y <= 53.5) & (ball_start_end.ball_end_y >= -0.5)]

        # merge tracking_df with ball_end and ball_start
        tracking_df = tracking_df.loc[tracking_df.nflId != 0].merge(ball_start_end, on=['gameId', 'playId'])
        print('merged with ball', time.time() - start_time)

        if self.tuning == TuningParam.sigma:
            # for each player, label whether they reached the ball (radius of 1.5 yds)
            self.player_reached = tracking_df.loc[tracking_df.event == 'pass_arrived'][['uniqueId',
                'nflId', 'x', 'y', 'ball_end_x', 'ball_end_y', 'pass_outcome', 'team_pos']].copy()
            self.player_reached['close_to_ball'] = np.less_equal(np.linalg.norm(np.stack([self.player_reached.x.values,
                        self.player_reached.y.values], axis=-1) - np.stack([self.player_reached.ball_end_x.values,
                        self.player_reached.ball_end_y.values], axis=-1), axis=1), 1.5).astype(int)

        elif self.tuning == TuningParam.lamb:
            self.player_reached = tracking_df.loc[tracking_df.event == 'pass_arrived'][['uniqueId',
                'nflId', 'team_pos', 'x', 'y', 'ball_end_x', 'ball_end_y', 'pass_outcome']].copy()
            self.player_reached['close_to_ball'] = np.less_equal(np.linalg.norm(np.stack([self.player_reached.x.values,
                        self.player_reached.y.values], axis=-1) - np.stack([self.player_reached.ball_end_x.values,
                        self.player_reached.ball_end_y.values], axis=-1), axis=1), 1.5).astype(int)

            # remove frames where nobody is close to ball when ball arrives
            close_to_ball = self.player_reached.groupby('uniqueId').filter(lambda x: x.close_to_ball.sum() > 0)['uniqueId']
            tracking_df = tracking_df.loc[tracking_df.uniqueId.isin(close_to_ball)]
            self.player_reached = self.player_reached.loc[self.player_reached.uniqueId.isin(close_to_ball)]

            # TODO @SS I think this should be DEF here? good catch my b
            # control is given by (player is on defense) XOR (ball is caught)
            self.player_reached['control_ball'] = ((self.player_reached['team_pos'] == 'DEF') ^ \
                    self.player_reached['pass_outcome']).astype(int)

        elif self.tuning == TuningParam.av:
            # for each player, label whether they reached the ball (radius of 1.5 yds)
            self.player_reached = tracking_df.loc[tracking_df.event == 'pass_arrived'][['uniqueId',
                'nflId', 'team_pos', 'x', 'y', 'ball_end_x', 'ball_end_y']].copy()
            self.player_reached['dist_to_ball'] = np.linalg.norm(np.stack([self.player_reached.x.values,
                        self.player_reached.y.values], axis=-1) - np.stack([self.player_reached.ball_end_x.values,
                        self.player_reached.ball_end_y.values], axis=-1), axis=1)
            self.player_reached['closest_to_ball'] = self.player_reached.groupby(['uniqueId',
                'team_pos']).dist_to_ball.transform('min')
            self.player_reached['closest_to_ball'] = (self.player_reached['dist_to_ball'] == self.player_reached['closest_to_ball']).astype(int) \
                    * self.player_reached['dist_to_ball']
            self.player_reached = self.player_reached[['uniqueId', 'nflId', 'x', 'y', 'closest_to_ball']]
            print('computed closest to ball', time.time() - start_time)

        # replace positions with ints
        tracking_df = tracking_df.replace('OFF', 1)
        tracking_df = tracking_df.replace('DEF', 0)

        # drop excess columns
        if self.tuning == TuningParam.sigma or self.tuning == TuningParam.lamb:
            self.player_reached = self.player_reached.drop(columns=['pass_outcome', 'ball_end_x', 'ball_end_y', 'team_pos', 'x', 'y'])

        self.all_plays = tracking_df[['uniqueId', 'nflId', 'x', 'y', 'v_x', 'v_y',
            'a_x', 'a_y', 'team_pos', 'ball_start_x', 'ball_start_y', 'ball_end_x', 'ball_end_y']]

        # generate play list
        play_list = tracking_df.loc[(tracking_df.event == 'pass_forward') | (tracking_df.event == 'pass_arrived'), ['gameId', 'playId', 'event', 'first_frameId']].copy()
        play_list = play_list.replace({'pass_forward': 1, 'pass_arrived': 0})

        # calculate forward and arrived frame ids (to get rid of event field)
        play_list['forward_frameId'] = play_list['event'] * play_list['first_frameId']
        play_list['arrived_frameId'] = (1 - play_list['event']) * play_list['first_frameId']

        # aggregate frameIds
        play_list_grouped = play_list.groupby(['gameId', 'playId'])
        play_list['forward_frameId'] = play_list_grouped.forward_frameId.transform('max')
        play_list['arrived_frameId'] = play_list_grouped.arrived_frameId.transform('max')
        play_list = play_list.drop(columns=['event', 'first_frameId']).drop_duplicates()

        # calculate tof (units of 0.1 s)
        play_list['tof'] = np.clip(play_list['arrived_frameId'] - play_list['forward_frameId'], 1, 40)

        # turn play list into np array
        self.play_list = play_list.values

        # max number of players per play
        self.max_num = 17

    def __len__(self):
        return len(self.play_list)

    def __getitem__(self, idx):
        gameId = str(self.play_list[idx, 0])
        playId = str(self.play_list[idx, 1])
        forward_frameId = str(self.play_list[idx, 2])
        arrived_frameId = str(self.play_list[idx, 3])
        tof = self.play_list[idx, 4]

        # calculate unique ids
        forward_uniqueId = '_'.join([gameId, playId, forward_frameId])
        arrived_uniqueId = '_'.join([gameId, playId, arrived_frameId])

        # THIS MIGHT BE WHERE SLOWDOWN HAPPENS - TODO try and use np style indexing instead

        # load frame, sigma_label, and ball_end, only keep relevant frames
        frame = self.all_plays.loc[self.all_plays.uniqueId == forward_uniqueId]
        frame['tof'] = tof
        if self.tuning != TuningParam.alpha:
            sigma_lambda_label = self.player_reached.loc[self.player_reached.uniqueId == arrived_uniqueId].drop(columns='uniqueId')

        # generate data, label, fill missing data
        if self.tuning == TuningParam.lamb:
            nflIds = sigma_lambda_label.loc[sigma_lambda_label.close_to_ball == 1, 'nflId'].values
            data = torch.tensor(frame.loc[frame.nflId.isin(nflIds), ['nflId', 'x', 'y', 'v_x', 'v_y',
                'a_x', 'a_y', 'team_pos', 'ball_start_x', 'ball_start_y', 'ball_end_x', 'ball_end_y', 'tof']].values).float()
            label = torch.tensor(sigma_lambda_label.loc[sigma_lambda_label.close_to_ball == 1, 'control_ball'].values)
        elif self.tuning == TuningParam.sigma:
            data = torch.tensor(frame[['nflId', 'x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'team_pos',
                'ball_start_x', 'ball_start_y', 'ball_end_x', 'ball_end_y', 'tof']].values).float()
            label = torch.tensor(sigma_lambda_label['close_to_ball'].values)
        elif self.tuning == TuningParam.av:
            # create evaluate_dist column in frame to indicate whether each player was the closest to the ball
            sigma_lambda_label['evaluate_dist'] = (sigma_lambda_label['closest_to_ball'] > 0).astype(int)
            frame['evaluate_dist'] = sigma_lambda_label['evaluate_dist'].values
            data = torch.tensor(frame[['nflId', 'x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'team_pos',
                'ball_start_x', 'ball_start_y', 'evaluate_dist', 'ball_end_x', 'ball_end_y', 'tof']].values).float()
            label = torch.tensor(sigma_lambda_label[['x', 'y']].values * sigma_lambda_label.evaluate_dist.values.reshape(-1, 1))
        elif self.tuning == TuningParam.alpha:
            data = torch.tensor(frame[['nflId', 'x', 'y', 'v_x', 'v_y', 'a_x', 'a_y', 'team_pos',
                'ball_start_x', 'ball_start_y', 'ball_end_x', 'ball_end_y', 'tof']].values).float()
            # model gets prob assigned to true pass in frame, so we just need BCE(prob_true_pass, 1)
            label = torch.tensor(1)
        if data.size(0) < self.max_num:
            data = torch.cat([data, torch.zeros([self.max_num - data.size(0), data.size(1)])], dim=0)
            if self.tuning != TuningParam.alpha:
                # don't want to 0-pad label for alpha
                label = torch.cat([label, torch.zeros([self.max_num - label.size(0), *label.size()[1:]])], dim=0)

        # TODO(adit98) investigate why this happens, for now put this in as a hack
        if data.size(0) > self.max_num:
            data = data[:self.max_num]
            label = label[:self.max_num]

        return data, label

# Completion Probability Model
class CompProbModel(torch.nn.Module):
    def __init__(self, a_max=7.25, s_max=9.25, avg_ball_speed=20.0, tti_sigma=0.5, tti_lambda_off=1.0,
                    tti_lambda_def=1.0, ppc_alpha=1.0, tuning=None, use_ppc=False):
        super().__init__()
        # define self.tuning
        self.tuning = tuning

        # define parameters and whether or not to optimize
        self.tti_sigma = Parameter(torch.tensor([tti_sigma]),
                requires_grad=(self.tuning == TuningParam.sigma)).float()
        self.tti_lambda_off = Parameter(torch.tensor([tti_lambda_off]),
                requires_grad=(self.tuning == TuningParam.lamb)).float()
        self.tti_lambda_def = Parameter(torch.tensor([tti_lambda_def]),
                requires_grad=(self.tuning == TuningParam.lamb)).float()
        self.ppc_alpha = Parameter(torch.tensor([ppc_alpha]),
                requires_grad=(self.tuning == TuningParam.alpha)).float()
        self.a_max = Parameter(torch.tensor([a_max]), requires_grad= (self.tuning == TuningParam.av)).float()
        self.s_max = Parameter(torch.tensor([s_max]), requires_grad=(self.tuning == TuningParam.av)).float()
        self.reax_t = Parameter(torch.tensor([0.2])).float()
        self.avg_ball_speed = Parameter(torch.tensor([avg_ball_speed]), requires_grad=False).float()
        self.g = Parameter(torch.tensor([10.72468]), requires_grad=False) #y/s/s
        self.z_max = Parameter(torch.tensor([3.]), requires_grad=False)
        self.z_min = Parameter(torch.tensor([0.]), requires_grad=False)
        self.use_ppc = use_ppc
        self.zero_cuda = Parameter(torch.tensor([0.0], dtype=torch.float32), requires_grad=False)

        # define field grid
        self.x = torch.linspace(0.5, 119.5, 120).float()
        self.y = torch.linspace(-0.5, 53.5, 55).float()
        self.y[0] = -0.2
        self.yy, self.xx = torch.meshgrid(self.y, self.x)
        self.field_locs = Parameter(torch.flatten(torch.stack((self.xx, self.yy), dim=-1), end_dim=-2), requires_grad=False)  # (F, 2)
        self.T = Parameter(torch.linspace(0.1, 4, 40), requires_grad=False) # (T,)

        # for hist trans prob
        self.hist_x_min, self.hist_x_max = -9, 70
        self.hist_y_min, self.hist_y_max = -39, 40
        self.hist_t_min, self.hist_t_max = 10, 63
        self.T_given_Ls_df = pd.read_pickle('in/T_given_L.pkl')

    def get_hist_trans_prob(self, frame):
        B = len(frame)
        """ P(L|t) """
        ball_start = frame[:, 0, 8:10] # (B, 2)
        ball_start_ind = torch.round(ball_start).long()
        reach_vecs = self.field_locs.unsqueeze(0) - ball_start.unsqueeze(1)  # (B, F, 2)
        # mask for zeroing out parts of the field that are too far to be thrown to per the L_given_t model
        L_t_mask = torch.zeros(B, *self.xx.shape)  # (B, Y, X)
        b_zeros = torch.zeros(ball_start_ind.shape[0])
        b_ones = torch.ones(ball_start_ind.shape[0])
        for bb in range(B):
            L_t_mask[bb, max(0, ball_start_ind[bb,1]+self.hist_y_min):\
                        min(len(self.y)-1, ball_start_ind[bb,1]+self.hist_y_max),\
                     max(0, ball_start_ind[bb,0]+self.hist_x_min):\
                        min(len(self.x)-1, ball_start_ind[bb,0]+self.hist_x_max)] = 1.
        L_t_mask = L_t_mask.flatten(1)  # (B, F)
        L_given_t = L_t_mask #changed L_given_t to uniform after discussion
        # renormalize since part of L|t may have been off field
        L_given_t /= L_given_t.sum(1, keepdim=True)  # (B, F)

        """ P(T|L) """
        # we find T|L for sufficiently close spots (1 < L <= 60)
        reach_dist_int = torch.round(torch.linalg.norm(reach_vecs, dim=-1)).long()  # (B, F)
        reach_dist_in_bounds_idx = (reach_dist_int > 1) & (reach_dist_int <= 60)
        reach_dist_in_bounds = reach_dist_int[reach_dist_in_bounds_idx]  # 1d tensor
        T_given_L_subset = torch.from_numpy(self.T_given_Ls_df.set_index('pass_dist').loc[reach_dist_in_bounds, 'p'].to_numpy()).float()\
            .reshape(-1, len(self.T))  # (BF~, T) ; BF~ is subset of B*F that is in [1, 60] yds from ball
        T_given_L = torch.zeros(B*len(self.field_locs), len(self.T))  # (B, F, T)
        # fill in the subset of values computed above
        T_given_L[reach_dist_in_bounds_idx.flatten()] = T_given_L_subset
        T_given_L = T_given_L.reshape(B, len(self.field_locs), -1)  # (B, F, T)

        L_T_given_t = L_given_t[...,None] * T_given_L  # (B, F, T)
        L_T_given_t /= L_T_given_t.sum((1, 2), keepdim=True) # normalize all passes after some have been chopped off
        return L_T_given_t  # (B, F, T)

    def get_ppc_off(self, frame, p_int):
        assert self.use_ppc, 'Call made to get_ppc_off while use_ppc setting is False'
        B = frame.shape[0]
        J = p_int.shape[-1]
        ball_start = frame[:, 0, 8:10]  # (B, 2)
        player_teams = frame[:, :, 7]  # (B, J)
        reach_vecs = self.field_locs.unsqueeze(0) - ball_start.unsqueeze(1)  # B, F, 2
        # trajectory integration
        dx = reach_vecs[:, :, 0] #B, F
        dy = reach_vecs[:, :, 1] #B, F
        vx = dx[:, :, None]/self.T[None, None, :]   #F, T
        vy = dy[:, :, None]/self.T[None, None, :]   #F, T
        vz_0 = (self.T*self.g)/2                #T

        # note that idx (i, j, k) into below arrays is invalid when j < k
        traj_ts = self.T.repeat(len(self.field_locs), len(self.T), 1) #(F, T, T)
        traj_locs_x_idx = torch.round(torch.clip((ball_start[:, 0, None, None, None]+vx.unsqueeze(-1)*self.T), 0, len(self.x)-1)).int() # B, F, T, T
        traj_locs_y_idx = torch.round(torch.clip((ball_start[:, 1, None, None, None]+vy.unsqueeze(-1)*self.T), 0, len(self.y)-1)).int() # B, F, T, T
        traj_locs_z = 2.0+vz_0.view(1, -1, 1)*traj_ts-0.5*self.g*traj_ts*traj_ts #F, T, T
        lambda_z = torch.where((traj_locs_z<self.z_max) & (traj_locs_z>self.z_min), 1, 0) #F, T, T
        path_idxs = (traj_locs_y_idx * self.x.shape[0] + traj_locs_x_idx).long().reshape(B, -1)  # (B, F*T*T)
        # 10*traj_ts - 1 converts the times into indices - hacky
        traj_t_idxs = (10*traj_ts - 1).long().repeat(B, 1, 1, 1).reshape(B, -1)  # (B, F*T*T)
        p_int_traj = torch.stack([p_int[bb, path_idxs[bb], traj_t_idxs[bb], :] for bb in range(B)])\
                        .reshape(*traj_locs_x_idx.shape, -1) * lambda_z.unsqueeze(-1)  # B, F, T, T, J
        p_int_traj_sum = p_int_traj.sum(dim=-1, keepdim=True)  # B, F, T, T, J
        norm_factor = torch.maximum(torch.ones_like(p_int_traj_sum), p_int_traj_sum)  # B, F, T, T
        p_int_traj_norm = p_int_traj / norm_factor  # B, F, T, T, J

        # independent int probs at each point on trajectory
        all_p_int_traj = torch.sum(p_int_traj_norm, dim=-1)  # B, F, T, T
        # off_p_int_traj = torch.sum((player_teams == 1)[:,None,None,None] * p_int_traj_norm, dim=-1)  # B, F, T, T
        # def_p_int_traj = torch.sum((player_teams == 0)[:,None,None,None] * p_int_traj_norm, dim=-1)  # B, F, T, T
        ind_p_int_traj = p_int_traj_norm #use for analyzing specific players; # B, F, T, T, J

        # calc decaying residual probs after you take away p_int on earlier times in the traj
        compl_all_p_int_traj = 1-all_p_int_traj  # B, F, T, T
        remaining_compl_p_int_traj = torch.cumprod(compl_all_p_int_traj, dim=-1)  # B, F, T, T
        # maximum 0 because if it goes negative the pass has been caught by then and theres no residual probability
        shift_compl_cumsum = torch.roll(remaining_compl_p_int_traj, 1, dims=-1)  # B, F, T, T
        shift_compl_cumsum[:, :, :, 0] = 1

        # multiply residual prob by p_int at that location and lambda
        lambda_all = self.tti_lambda_off * player_teams + self.tti_lambda_def * (1 - player_teams)  # B, J
        # off_completion_prob_dt = shift_compl_cumsum * off_p_int_traj  # B, F, T, T
        # def_completion_prob_dt = shift_compl_cumsum * def_p_int_traj  # B, F, T, T
        # all_completion_prob_dt = off_completion_prob_dt + def_completion_prob_dt  # B, F, T, T
        ind_completion_prob_dt = shift_compl_cumsum.unsqueeze(-1) * ind_p_int_traj  # F, T, T, J

        # now accumulate values over total traj for each team and take at T=t
        # all_completion_prob = torch.cumsum(all_completion_prob_dt, dim=-1)  # B, F, T, T
        # off_completion_prob = torch.cumsum(off_completion_prob_dt, dim=-1)  # B, F, T, T
        # def_completion_prob = torch.cumsum(def_completion_prob_dt, dim=-1)  # B, F, T, T
        ind_completion_prob = torch.cumsum(ind_completion_prob_dt, dim=-2)  # B, F, T, T, J

        # this einsum takes the diagonal values over the last two axes where T = t
        # this takes care of the t > T issue.
        # ppc_all = torch.einsum('...ii->...i', all_completion_prob)  # B, F, T
        # ppc_off = torch.einsum('...ii->...i', off_completion_prob)  # B, F, T
        # ppc_def = torch.einsum('...ii->...i', def_completion_prob)  # B, F, T
        ppc_ind = torch.einsum('...iij->...ij', ind_completion_prob)  # B, F, T, J
        ppc_ind *= lambda_all[:,None,None,:]
        # no_p_int_pass = 1-ppc_all  # B, F, T

        ppc_off = torch.sum(ppc_ind * player_teams[:,None,None,:], dim=-1)  # B, F, T
        ppc_def = torch.sum(ppc_ind * (1-player_teams)[:,None,None,:], dim=-1)  # B, F, T

        # assert torch.allclose(all_p_int_pass, off_p_int_pass + def_p_int_pass, atol=0.01)
        # assert torch.allclose(all_p_int_pass, ind_p_int_pass.sum(-1), atol=0.01)
        # return off_p_int_pass, def_p_int_pass, ind_p_int_pass
        return ppc_off, ppc_def, ppc_ind

    def forward(self, frame):
        v_x_r = frame[:, :, 5] * self.reax_t + frame[:, :, 3]
        v_y_r = frame[:, :, 6] * self.reax_t + frame[:, :, 4]
        v_r_mag = torch.norm(torch.stack([v_x_r, v_y_r], dim=-1), dim=-1)
        v_r_theta = torch.atan2(v_y_r, v_x_r)

        x_r = frame[:, :, 1] + frame[:, :, 3] * self.reax_t + 0.5 * frame[:, :, 5] * self.reax_t**2
        y_r = frame[:, :, 2] + frame[:, :, 4] * self.reax_t + 0.5 * frame[:, :, 6] * self.reax_t**2

        # get each player's team, location, and velocity
        player_teams = frame[:, :, 7] # B, J
        reaction_player_locs = torch.stack([x_r, y_r], dim=-1) # (J, 2)
        reaction_player_vels = torch.stack([v_x_r, v_y_r], dim=-1) #(J, 2)

        # calculate each player's distance from each field location
        int_d_vec = self.field_locs.unsqueeze(1).unsqueeze(0) - reaction_player_locs.unsqueeze(1) #F, J, 2
        int_d_mag = torch.norm(int_d_vec, dim=-1) # F, J
        int_d_theta = torch.atan2(int_d_vec[..., 1], int_d_vec[..., 0]) # F, J

        # take dot product of velocity and direction
        int_s0 = torch.clamp(torch.sum(int_d_vec * reaction_player_vels.unsqueeze(1), dim=-1) / int_d_mag,
                -1 * self.s_max.item(), self.s_max.item()) #F, J

        # calculate time it takes for each player to reach each field position accounting for their current velocity and acceleration
        t_lt_smax = (self.s_max - int_s0) / self.a_max  #F, J,
        d_lt_smax = t_lt_smax * ((int_s0 + self.s_max) / 2) #F, J,

        # if accelerating would overshoot, then t = -v0/a + sqrt(v0^2/a^2 + 2x/a) (from kinematics)
        t_lt_smax = torch.where(d_lt_smax > int_d_mag, -int_s0 / self.a_max + \
                torch.sqrt((int_s0 / self.a_max) ** 2 + 2 * int_d_mag / self.a_max), t_lt_smax) # F, J
        d_lt_smax = torch.max(torch.min(d_lt_smax, int_d_mag), torch.zeros_like(d_lt_smax)) # F, J

        d_at_smax = int_d_mag - d_lt_smax               #F, J,
        t_at_smax = d_at_smax / self.s_max              #F, J,
        t_tot = self.reax_t + t_lt_smax + t_at_smax     # F, J,

        # get true pass (tof and ball_end) to tune on (subtract 1 from tof, add 1 to y for correct indexing)
        tof = torch.round(frame[:, 0, -1]).long().view(-1, 1, 1, 1).repeat(1, t_tot.size(1), 1, t_tot.size(-1)) - 1

        # ball ind
        ball_end_x = frame[:, 0, -3].int()
        ball_end_y = frame[:, 0, -2].int() + 1
        ball_field_ind = (ball_end_y * self.x.shape[0] + ball_end_x).long().view(-1, 1, 1).repeat(1, 1, t_tot.size(-1))

        if self.tuning == TuningParam.av:
            # collapse extra dims
            tof = self.T[tof[:, 0, 0, 0]].float()

            # select field in for all the position and velocity values calculated previously
            t_lt_smax = torch.gather(t_lt_smax, 1, ball_field_ind).squeeze() # J,
            d_lt_smax = torch.gather(d_lt_smax, 1, ball_field_ind).squeeze() # J,
            d_at_smax = torch.gather(d_at_smax, 1, ball_field_ind).squeeze()
            t_at_smax = torch.gather(t_at_smax, 1, ball_field_ind).squeeze()
            t_tot = torch.gather(t_tot, 1, ball_field_ind).squeeze()
            int_s0 = torch.gather(int_s0, 1, ball_field_ind).squeeze()

            int_d_theta = torch.gather(int_d_theta, 1, ball_field_ind).squeeze()
            int_d_mag = torch.gather(int_d_mag, 1, ball_field_ind).squeeze()

            # projected locations at t = tof, f = ball_field_ind
            d_proj = torch.where(tof.unsqueeze(-1) <= self.reax_t, self.zero_cuda,
                    torch.where(tof.unsqueeze(-1) <= (t_lt_smax + self.reax_t),
                    (int_s0 * (tof.unsqueeze(-1) - self.reax_t)) + 0.5 * self.a_max \
                            * (tof.unsqueeze(-1) - self.reax_t) ** 2,
                    torch.where(tof.unsqueeze(-1) <= (t_lt_smax + t_at_smax + self.reax_t),
                    (d_lt_smax + (d_at_smax * (tof.unsqueeze(-1) - t_lt_smax - self.reax_t))),
                    int_d_mag))) # J,

            d_proj = torch.minimum(d_proj, int_d_mag)

            x_proj = reaction_player_locs[..., 0]  + d_proj * torch.cos(int_d_theta)  # J
            y_proj = reaction_player_locs[..., 1]  + d_proj * torch.sin(int_d_theta)  # J

            # mask x_proj and y_proj (only want loss on closest off and def players)
            player_mask = frame[:, :, -4]
            masked_x = player_mask * x_proj
            masked_y = player_mask * y_proj

            return torch.stack([masked_x, masked_y], dim=-1) # J, 2

        # subtract the arrival time (t_tot) from time of flight of ball
        int_dT = self.T.view(1, 1, -1, 1) - t_tot.unsqueeze(2)         #F, T, J

        # calculate interception probability for each player, field loc, time of flight (logistic function)
        p_int = torch.sigmoid((3.14 / (1.732 * self.tti_sigma)) * int_dT) # (B, F, T, J)

        if self.tuning == TuningParam.sigma:
            p_int = torch.gather(p_int, 2, tof).squeeze()
            p_int = torch.gather(p_int, 1, ball_field_ind).squeeze()
            return p_int

        elif self.tuning == TuningParam.alpha:
            h_trans_prob = self.get_hist_trans_prob(frame)  # (B, F, T)
            if self.use_ppc:
                ppc_off, *_ = self.get_ppc_off(frame, p_int)
                trans_prob = h_trans_prob * torch.pow(ppc_off, self.ppc_alpha)  # (B, F, T)
            else:
                # p_int summed over all offensive players
                p_int_off = torch.sum(p_int * (player_teams == 1), dim=-1)  # (B, F, T)
                trans_prob = h_trans_prob * torch.pow(p_int_off, self.ppc_alpha)  # (B,)
            trans_prob /= trans_prob.sum(dim=(1, 2), keepdim=True)  # (B, F, T)
            # index into true pass. [...,0] necessary on indices because no J dimension
            trans_prob_throw = torch.gather(trans_prob, 2, tof[...,0]).squeeze()
            trans_prob_throw = torch.gather(trans_prob_throw, 1, ball_field_ind[...,0]).squeeze()  # (B,)
            return trans_prob_throw

        elif self.tuning == TuningParam.lamb:
            assert self.use_ppc, 'need to use ppc to tune lambda'
            *_, ppc_ind = self.get_ppc_off(frame, p_int)  # ppc_ind: (B, F, T, J)
            ppc_ind_throw = torch.gather(ppc_ind, 2, tof).squeeze()  # B, F, J
            ppc_ind_throw = torch.gather(ppc_ind_throw, 1, ball_field_ind).squeeze()  # B, J
            return ppc_ind_throw

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default=os.path.join(get_repo_dir(), 'data'), help='path to directory with data files')
    parser.add_argument('-j', '--num_workers', default=4, help='number of data loader workers', type=int)
    parser.add_argument('-b', '--batch_size', default=4, help='batch size to use', type=int)
    parser.add_argument('-e', '--epochs', default=5, help='number of training epochs', type=int)
    parser.add_argument('-t', '--tuning', default=None, help='parameter to tune (None if running in eval)')
    parser.add_argument('-s', '--split', default='1', help='week number to run on or all')
    parser.add_argument('-p', '--ppc', default=True, help='whether to use ppc (only relevant for tuning alpha)')
    args = parser.parse_args()

    # Initialize Dataset, Model and Run Training Loop
    training = True
    if args.tuning is None:
        training = False
        event_filter = 'pass_forward'
    elif args.tuning == 'sigma':
        TUNING = TuningParam.sigma
        event_filter = 'pass_forward'
    elif args.tuning == 'lambda':
        TUNING = TuningParam.lamb
        event_filter = 'pass_forward'
    elif args.tuning == 'alpha':
        TUNING = TuningParam.alpha
        event_filter = 'pass_forward'
    elif args.tuning == 'av':
        TUNING = TuningParam.av
    else:
        raise NotImplementedError("Tuning " + args.tuning + " is not supported.")

    if args.split == 'all':
        all_weeks = True
        wk = None
    else:
        all_weeks = False
        wk = args.split
    ds = PlaysDataset(data_dir=args.data_dir, wk=wk, all_weeks=all_weeks, tuning=TUNING)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)

    model = CompProbModel(tti_sigma=0.5, tuning=TUNING, use_ppc=args.ppc)
    if TUNING == TuningParam.av:
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = torch.nn.BCELoss()

    # check if we want cuda
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    optimizer = torch.optim.Adam(model.parameters())
    total_loss = 0

    if not training:
        model.eval()
        epochs = 1
    else:
        epochs = args.epochs

    for epoch in range(1, epochs + 1):
        prog_bar = tqdm(loader)
        total_loss = 0
        for ind, (data, target) in enumerate(prog_bar):
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = model(data)
            loss = loss_fn(output, target.float())
            total_loss = total_loss + loss.detach().cpu().item()

            # step gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prog_bar.set_description("Batch %d Loss %.3f" % (epoch, total_loss / (ind + 1)))

        # save model
        torch.save(model.state_dict(), 'tuned_model.pt')

    print('a_max', model.a_max)
    print('s_max', model.s_max)
    print(model.tti_lambda_off)
    print(model.tti_lambda_def)
    print(model.tti_sigma)
    print(model.ppc_alpha)

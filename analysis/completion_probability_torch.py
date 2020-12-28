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

class TuningParam(Enum):
    sigma = 1
    lamb = 2

class PlaysDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, wk=1, all_weeks=False, event_filter=None, tuning=None):
        # time code TODO(adit98) remove this later
        self.tuning = tuning

        if all_weeks:
            all_data = []
            for week in range(5, 10):
                all_data.append(pd.read_csv(os.path.join(data_dir, 'week%d_norm.csv' % week)))
                
            tracking_df = pd.concat(all_data)
            
        else:
            # load csvs
            tracking_df = pd.read_csv(os.path.join(data_dir, 'week%s_norm.csv' % wk))

        # remove frames with more than 17 players' tracking data + QB + ball (19)
        tracking_df = tracking_df.groupby(['gameId', 'playId', 'frameId']).filter(lambda x: len(x.nflId.unique()) <= 19)

        # get valid frames for tuning from tracking df (consider every pass, labels are 1 if there is a player close by)
        pass_forward_plays = tracking_df.loc[tracking_df['event'] == 'pass_forward'][['gameId', 'playId']].drop_duplicates()
        pass_attempted_plays = tracking_df.loc[tracking_df['event'] == 'pass_arrived'][['gameId', 'playId']].drop_duplicates()
        tracking_df = pass_forward_plays.merge(pass_attempted_plays.merge(tracking_df, on=['gameId', 'playId'], how='inner'), on=['gameId', 'playId'], how='inner')

        # add forward_frameId, arrived_frameId to play_list which contains first frame of pass_forward, pass_arrived for each play
        tracking_df['forward_frameId'] = tracking_df.loc[tracking_df.event == 'pass_forward'].groupby(['gameId', 'playId']).frameId.transform('min')
        tracking_df['arrived_frameId'] = tracking_df.loc[tracking_df.event == 'pass_arrived'].groupby(['gameId', 'playId']).frameId.transform('min')

        # calculate ball ending position
        ball_end = tracking_df.loc[(tracking_df.nflId == 0) & (tracking_df.event == 'pass_arrived')][['gameId', 'playId', 'x', 'y']].copy()
        ball_end = ball_end.rename(columns={'x': 'ball_end_x', 'y': 'ball_end_y'}).drop_duplicates()
        
        # calculate ball position at throw
        ball_start = tracking_df.loc[(tracking_df.nflId == 0) & (tracking_df.event == 'pass_forward')][['gameId', 'playId', 'x', 'y']].copy()
        ball_start = ball_start.rename(columns={'x': 'ball_start_x', 'y': 'ball_start_y'}).drop_duplicates()

        # merge into single df
        ball_start_end = ball_end.merge(ball_start, on=['gameId', 'playId'])

        # remove plays where ball is thrown out of bounds
        ball_start_end = ball_start_end.loc[(ball_start_end.ball_end_x <= 119.5) & (ball_start_end.ball_end_x >= 0.5) & \
                (ball_start_end.ball_end_y <= 53.5) & (ball_start_end.ball_end_y >= -0.5)]
        
        # merge tracking_df with ball_end and ball_start
        tracking_df = tracking_df.loc[tracking_df.nflId != 0].merge(ball_start_end, on=['gameId', 'playId'])

        # this shit is fucking retarded why do you have to copy this
        play_list = tracking_df[['gameId', 'playId', 'forward_frameId', 'arrived_frameId']].copy()
        play_list_grouped = play_list.groupby(['gameId', 'playId'])
        play_list['forward_frameId'] = play_list_grouped.forward_frameId.transform('mean')
        play_list['arrived_frameId'] = play_list_grouped.arrived_frameId.transform('mean')
        play_list['tof'] = np.clip(play_list['arrived_frameId'] - play_list['forward_frameId'], 0, 40)
        play_list = play_list.drop_duplicates()

        if self.tuning == TuningParam.sigma:
            # for each player, label whether they reached the ball (radius of 1.5 yds)
            self.player_reached = tracking_df.loc[tracking_df.event == 'pass_arrived'][['gameId', 'playId',
                'frameId', 'nflId', 'team_pos', 'x', 'y', 'ball_end_x', 'ball_end_y']].copy()
            self.player_reached['close_to_ball'] = np.less_equal(np.linalg.norm(np.stack([self.player_reached.x.values,
                        self.player_reached.y.values], axis=-1) - np.stack([self.player_reached.ball_end_x.values,
                        self.player_reached.ball_end_y.values], axis=-1), axis=1), 1.5).astype(int)

        elif self.tuning == TuningParam.lamb:
            self.player_reached = tracking_df.loc[tracking_df.event == 'pass_arrived'][['gameId', 'playId',
                'frameId', 'nflId', 'team_pos', 'x', 'y', 'ball_end_x', 'ball_end_y']].copy()
            self.player_reached['close_to_ball'] = np.less_equal(np.linalg.norm(np.stack([self.player_reached.x.values,
                        self.player_reached.y.values], axis=-1) - np.stack([self.player_reached.ball_end_x.values,
                        self.player_reached.ball_end_y.values], axis=-1), axis=1), 1.5).astype(int)

            # remove frames where nobody is close to ball when ball arrives
            close_to_ball = self.player_reached.groupby(['gameId', 'playId']).filter(lambda x: x.close_to_ball.sum() > 0)[['gameId',
                'playId']].copy().drop_duplicates()
            tracking_df = close_to_ball.merge(tracking_df, on=['gameId', 'playId'])
            self.player_reached = close_to_ball.merge(self.player_reached, on=['gameId', 'playId'])
            play_list = tracking_df[['gameId', 'playId']].copy()

            # TODO @SS I think this should be DEF here?
            # control is given by (player is on offense) XOR (ball is caught)
            self.player_reached['control_ball'] = ((self.player_reached['team_pos'] == 'DEF') ^ \
                    self.player_reached['event'].isin(['pass_outcome_caught', 'pass_outcome_touchdown'])).astype(int)

        # replace positions with ints
        self.player_reached = self.player_reached.replace('OFF', 1)
        self.player_reached = self.player_reached.replace('DEF', 0)

        # replace positions with ints
        tracking_df = tracking_df.replace('OFF', 1)
        tracking_df = tracking_df.replace('DEF', 0)

        # calculate tracking_df/player_reached inds for each play
        #event_ends = tracking_df.groupby(['gameId', 'playId', 'frameId'])

        # drop duplicate columns and store tracking_df
        self.player_reached = self.player_reached.drop(columns=['ball_end_x', 'ball_end_y'])
        self.all_plays = tracking_df

        # turn play list into np array
        self.play_list = play_list.values

        # max number of players per play
        self.max_num = 17

    def __len__(self):
        return len(self.play_list)
    
    def __getitem__(self, idx):
        gameId = self.play_list[idx, 0]
        playId = self.play_list[idx, 1]
        forward_frameId = self.play_list[idx, 2]
        arrived_frameId = self.play_list[idx, 3]
        tof = self.play_list[idx, 4]
        # TODO(adit98) calculate play_start_ind, play_end_ind, remove gameId, playId
        #play_start_ind = 

        # THIS MIGHT BE WHERE SLOWDOWN HAPPENS - try and use np style indexing instead
        # frame = self.all_plays.iloc[play_start_ind:play_end_ind + 1]
        # load frame, sigma_label, and ball_end, only keep relevant frames
        frame = self.all_plays.loc[(self.all_plays.gameId == gameId) & (self.all_plays.playId == playId) & (self.all_plays.frameId == forward_frameId)]
        frame['tof'] = tof

        if self.tuning == TuningParam.lamb:
            sigma_lambda_label = self.player_reached.loc[(self.player_reached.gameId == gameId) & (self.player_reached.playId == playId) & \
                (self.player_reached.frameId == arrived_frameId)][['nflId', 'close_to_ball', 'control_ball']].copy()
        else:
            sigma_lambda_label = self.player_reached.loc[(self.player_reached.gameId == gameId) & (self.player_reached.playId == playId) & \
                    (self.all_plays.frameId == arrived_frameId)][['nflId', 'close_to_ball']].copy()

        ## this is used to pick which frame we want our PPCF to be calculated based off of
        #if self.event_filter is not None:
        #    frame = frame[frame.event == self.event_filter]

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

        if data.size(0) < self.max_num:
            data = torch.cat([data, torch.ones([self.max_num - data.size(0), data.size(1)])], dim=0)
            label = torch.cat([label, torch.zeros([self.max_num - label.size(0)])], dim=0)

        # TODO(adit98) investigate why this happens, for now put this in as a hack
        if data.size(0) > self.max_num:
            data = data[:self.max_num]
            label = label[:self.max_num]

        return data, label.long()

# Completion Probability Model
class CompProbModel(torch.nn.Module):
    def __init__(self, a_max=7.0, s_max=9.0, avg_ball_speed=20.0, tti_sigma=0.5, tti_lambda_off=1.0, tti_lambda_def=1.0, tuning=None):
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
        self.a_max = Parameter(torch.tensor([a_max]), requires_grad=True).float()
        self.s_max = Parameter(torch.tensor([s_max]), requires_grad=True).float()
        self.reax_t = Parameter(self.s_max / self.a_max, requires_grad=False).float()
        self.avg_ball_speed = Parameter(torch.tensor([avg_ball_speed])).float()
        self.g = Parameter(torch.tensor([10.72468]), requires_grad=False) #y/s/s
        self.z_max = Parameter(torch.tensor([3.]), requires_grad=False)
        self.z_min = Parameter(torch.tensor([0.]), requires_grad=False)

        # define field grid
        self.x = torch.linspace(0.5, 119.5, 120)
        self.y = torch.linspace(-0.5, 53.5, 55)
        self.y[0] = -0.2
        self.yy, self.xx = torch.meshgrid(self.y, self.x)
        self.field_locs = Parameter(torch.flatten(torch.stack((self.xx, self.yy), dim=-1), end_dim=-2), requires_grad=False)  # (F, 2)
        self.T = Parameter(torch.linspace(0.1, 4, 40), requires_grad=False) # (T,)


    def forward(self, frame):
        v_x_r = frame[:, :, 5] * self.reax_t + frame[:, :, 3]
        v_y_r = frame[:, :, 6] * self.reax_t + frame[:, :, 4]
        v_r_mag = torch.norm(torch.stack([v_x_r, v_y_r], dim=-1), dim=-1)
        v_r_theta = torch.atan2(v_y_r, v_x_r)
        
        x_r = frame[:, :, 1] + frame[:, :, 3] * self.reax_t + 0.5 * frame[:, :, 5] * self.reax_t**2
        y_r = frame[:, :, 2] + frame[:, :, 4] * self.reax_t + 0.5 * frame[:, :, 6] * self.reax_t**2
        
        # get each player's team, location, and velocity
        player_teams = frame[:, :, 7] # B, J
        reaction_player_locs = torch.stack([x_r, y_r], dim=-1).int() # (J, 2)
        reaction_player_vels = torch.stack([v_x_r, v_y_r], dim=-1) #(J, 2)
        
        # calculate each player's distance from each field location
        int_d_vec = self.field_locs.unsqueeze(1).unsqueeze(0) - reaction_player_locs.unsqueeze(1) #F, J, 2
        int_d_mag = torch.norm(int_d_vec, dim=-1) # F, J
        
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

        # subtract the arrival time (t_tot) from time of flight of ball
        int_dT = self.T.view(1, 1, -1, 1) - t_tot.unsqueeze(2)         #F, T, J
        
        # calculate interception probability for each player, field loc, time of flight (logistic function)
        p_int = torch.sigmoid((3.14 / (1.732 * self.tti_sigma)) * int_dT) #F, T, J

        # get true pass (tof and ball_end) to tune on (subtract 1 from tof, add 1 to y for correct indexing)
        tof = torch.round(frame[:, 0, -1]).long().view(-1, 1, 1, 1).repeat(1, p_int.size(1), 1, p_int.size(-1)) - 1

        ball_end_x = frame[:, 0, -3].int()
        ball_end_y = frame[:, 0, -2].int() + 1

        ball_field_ind = (ball_end_y * self.x.shape[0] + ball_end_x).long().view(-1, 1, 1).repeat(1, 1, p_int.size(-1))

        if self.tuning == TuningParam.sigma:
            p_int = torch.gather(p_int, 2, tof).squeeze()
            p_int = torch.gather(p_int, 1, ball_field_ind).squeeze()
            return p_int

        else:
            # get ball_start
            ball_start = frame[:, 0, 8:10]
            reach_vecs = self.field_locs - ball_start
            reach_dist = torch.norm(reach_vecs, dim=-1)
        
            dx = reach_vecs[:, 0] #F
            dy = reach_vecs[:, 1] #F
            vx = dx[:, None]/self.T[None, :]   #F, T
            vy = dy[:, None]/self.T[None, :]   #F, T
            vz_0 = (self.T * self.g)/2    #T

            # note that idx (i, j, k) into below arrays is invalid when j < k
            traj_ts = self.T.repeat(len(self.field_locs), len(self.T), 1) #(F, T, T)
            traj_locs_x_idx = torch.round(torch.clamp((ball_start[0]+vx.unsqueeze(-1)*self.T),
                0, len(self.x)-1)).int() # F, T, T
            traj_locs_y_idx = torch.round(torch.clamp((ball_start[1]+vy.unsqueeze(-1)*self.T),
                0, len(self.y)-1)).int() # F, T, T
            traj_locs_z = 2.0+vz_0.view(1, -1, 1)*traj_ts-0.5*self.g*traj_ts*traj_ts #F, T, T
            lambda_z = torch.where((traj_locs_z<self.z_max) & (traj_locs_z>self.z_min), 1, 0) #F, T, T
            
            path_idxs = (traj_locs_y_idx * self.x.shape[0] + traj_locs_x_idx).flatten()  # (F*T*T,)
            # 10*traj_ts - 1 converts the times into indices - hacky
            traj_t_idxs = torch.round(10*traj_ts - 1).flatten().int()  # (F*T*T,)
            p_int_traj = p_int[:, path_idxs.long(), traj_t_idxs.long()].reshape((-1, *traj_locs_x_idx.shape,
                player_teams.shape[1])) * lambda_z.unsqueeze(-1) # B, F, T, T, J
            p_int_traj_sum = p_int_traj.sum(dim=-1)
            norm_factor = torch.maximum(torch.ones_like(p_int_traj_sum).float(), p_int_traj_sum)  # B, F, T, T
            p_int_traj_norm = (p_int_traj / norm_factor.unsqueeze(-1))  # B, F, T, T, J
            
            # independent int probs at each point on trajectory
            all_p_int_traj = torch.sum(p_int_traj_norm, dim=-1)  # B, F, T, T
            off_p_int_traj = torch.sum((player_teams == 1)[:,None,None,None] * p_int_traj_norm, dim=-1)  # B, F, T, T
            def_p_int_traj = torch.sum((player_teams == 0)[:,None,None,None] * p_int_traj_norm, dim=-1)  # B, F, T, T
            ind_p_int_traj = p_int_traj_norm #use for analyzing specific players; # B, F, T, T, J
            
            # calc decaying residual probs after you take away p_int on earlier times in the traj 
            compl_all_p_int_traj = 1-all_p_int_traj  # B, F, T, T
            remaining_compl_p_int_traj = torch.cumprod(compl_all_p_int_traj, dim=-1)  # B, F, T, T
            # maximum 0 because if it goes negative the pass has been caught by then and theres no residual probability
            shift_compl_cumsum = torch.roll(remaining_compl_p_int_traj, 1, dims=-1)  # B, F, T, T
            shift_compl_cumsum[:, :, 0] = 1
            
            # multiply residual prob by p_int at that location and lambda
            lambda_all = self.tti_lambda_off * player_teams + self.tti_lambda_def * (1 - player_teams)  # B, J
            off_completion_prob_dt = shift_compl_cumsum * off_p_int_traj * self.tti_lambda_off  # B, F, T, T
            def_completion_prob_dt = shift_compl_cumsum * def_p_int_traj * self.tti_lambda_def  # B, F, T, T
            all_completion_prob_dt = off_completion_prob_dt + def_completion_prob_dt  # B, F, T, T
            ind_completion_prob_dt = shift_compl_cumsum.unsqueeze(-1) * ind_p_int_traj * lambda_all[:,None,None,None]  # F, T, T, J
            
            # now accumulate values over total traj for each team and take at T=t
            all_completion_prob = torch.cumsum(all_completion_prob_dt, dim=-1)  # B, F, T, T
            off_completion_prob = torch.cumsum(off_completion_prob_dt, dim=-1)  # B, F, T, T
            def_completion_prob = torch.cumsum(def_completion_prob_dt, dim=-1)  # B, F, T, T
            ind_completion_prob = torch.cumsum(ind_completion_prob_dt, dim=-2)  # B, F, T, T, J
            
            # this einsum takes the diagonal values over the last two axes where T = t
            # this takes care of the t > T issue.
            all_p_int_pass = torch.einsum('...ii->...i', all_completion_prob)  # B, F, T
            off_p_int_pass = torch.einsum('...ii->...i', off_completion_prob)  # B, F, T
            def_p_int_pass = torch.einsum('...ii->...i', def_completion_prob)  # B, F, T
            ind_p_int_pass = torch.einsum('...iij->...ij', ind_completion_prob)  # B, F, T, J
            no_p_int_pass = 1-all_p_int_pass  # B, F, T

            assert torch.allclose(all_p_int_pass, off_p_int_pass + def_p_int_pass, atol=0.01)
            assert torch.allclose(all_p_int_pass, ind_p_int_pass.sum(-1), atol=0.01)
            # return off_p_int_pass, def_p_int_pass, ind_p_int_pass
            ret_val = torch.gather(ind_p_int_pass, 2, tof).squeeze()  # B, F, J
            ret_val = torch.gather(ret_val, 1, ball_field_ind).squeeze()  # B, J
            return ret_val

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='../data', help='path to directory with data files')
    parser.add_argument('-j', '--num_workers', default=4, help='number of data loader workers', type=int)
    parser.add_argument('-b', '--batch_size', default=4, help='batch size to use', type=int)
    parser.add_argument('-e', '--epochs', default=5, help='number of training epochs', type=int)
    parser.add_argument('-t', '--tuning', default=None, help='parameter to tune (None if running in eval)')
    parser.add_argument('-s', '--split', default='1', help='week number to run on or all')
    parser.add_argument('-w', '--reg_weight', default=0.1, help='regularization weight', type=int)
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
    else:
        raise NotImplementedError("Tuning " + args.tuning + " is not supported.")

    if args.split == 'all':
        all_weeks = True
        wk = None
    else:
        all_weeks = False
        wk = args.split
    ds = PlaysDataset(data_dir=args.data_dir, wk=wk, all_weeks=all_weeks, event_filter=event_filter, tuning=TUNING)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)

    model = CompProbModel(tti_sigma=0.5, tuning=TUNING)
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

            # TODO(adit98) remove if this doesn't work properly (regularizing sigma magnitude)
            loss = (1 - args.reg_weight) * loss_fn(output, target.float()) + args.reg_weight * torch.abs(model.tti_sigma)
            total_loss = total_loss + loss.detach().cpu().item()

            # step gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prog_bar.set_description("Batch %d Loss %.3f" % (epoch, total_loss / (ind + 1)))

        # save model
        torch.save(model.state_dict(), 'tuned_model.pt')

    print(model.tti_lambda_off)
    print(model.tti_lambda_def)
    print(model.tti_sigma)

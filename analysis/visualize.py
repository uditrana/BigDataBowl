# Utility Libraries
from datetime import datetime
import pytz
import pandas as pd

# Computation Libraries
import numpy as np

# Plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

from matplotlib.path import Path

colors = pd.read_csv(
    'https://raw.githubusercontent.com/uditrana/NFLFastR_analysis/master/nfl_colors.tsv', delimiter='\t')

# Animation
# Animation Class
# First let us make an animation that will show all of the entities moving across the pitch over time through the play. We are in luck that we have timestamps for each of the timepoints when the measurement was taken. This allows us to make a very precise estimate as to what the sampling rate is, and thus we can set an accurate framerate for the animation to simulate the play at "real time". This is calculated on line 26 as self._mean_interval_ms.

# In this animation we plot:

# Player positions
# Line of scrimmage
# Ball
# Velocity vectors of the players
# Orientation of players
# Player last name
# Player position
# Player number
# On small screens this may be too much information, but this just shows how you can do it. Feel free to remove as you wish. I will not go into this animation class in too much detail as it is not the focus of this notebook. Feel free to go through it, it is mostly just using matplotlib and basic math to show what we want.


class AnimatePlay:
    def __init__(self, play_df, plot_size_len) -> None:
        """Initializes the datasets used to animate the play.

        Parameters
        ----------
        play_df : DataFrame
            Dataframe corresponding to the play information for the play that requires
            animation. This data will come from the weeks dataframe and contains position
            and velocity information for each of the players and the football.

        Returns
        -------
        None
        """
        self._MAX_FIELD_Y = 53.3
        self._MAX_FIELD_X = 120
        self._MAX_FIELD_PLAYERS = 22

        # self._CPLT = sns.color_palette("husl", 2)
        self._offense_color = colors.loc[colors.team == play_df.loc[play_df.team_pos == 'OFF']['teamAbbr'].iloc[0]]
        self._defense_color = colors.loc[colors.team == play_df.loc[play_df.team_pos == 'DEF']['teamAbbr'].iloc[0]]
        self._frame_data = play_df
        self._game_id, self._play_id = play_df.iloc[:1][['gameId', 'playId']].to_records(index=False)[0]
        self._times = sorted(play_df.time.unique())
        self._stream = self.data_stream()

        self._date_format = "%Y-%m-%dT%H:%M:%S.%fZ"
        self._mean_interval_ms = np.mean(
            [delta.microseconds / 1000
             for delta in np.diff(
                 np.array(
                     [pytz.timezone('US/Eastern').localize(datetime.strptime(date_string, self._date_format))
                      for date_string in self._times]))])

        self._fig = plt.figure(figsize=(plot_size_len, plot_size_len*(self._MAX_FIELD_Y/self._MAX_FIELD_X)))

        self._ax_field = plt.gca()

        self._ax_offense = self._ax_field.twinx()
        self._ax_defense = self._ax_field.twinx()
        self._ax_jersey = self._ax_field.twinx()

        self.ani = animation.FuncAnimation(
            self._fig, self.update, frames=len(self._times),
            interval=self._mean_interval_ms, init_func=self.setup_plot, blit=False)

        plt.close()

    @staticmethod
    def set_axis_plots(ax, max_x, max_y) -> None:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        ax.set_xlim([0, max_x])
        ax.set_ylim([0, max_y])

    @staticmethod
    def convert_orientation(x):
        return (-x + 90) % 360

    @staticmethod
    def polar_to_z(r, theta):
        return r * np.exp(1j * theta)

    @staticmethod
    def deg_to_rad(deg):
        return deg*np.pi/180

    def data_stream(self):
        for time in self._times:
            yield self._frame_data[self._frame_data.time == time]

    def setup_plot(self):
        self.set_axis_plots(self._ax_field, self._MAX_FIELD_X, self._MAX_FIELD_Y)

        # ball_snap_df = self._frame_data[(self._frame_data.event == 'ball_snap') & (self._frame_data.team == 'football')]
        self._ax_field.axvline(self._frame_data.iloc[0]['los'], color='k', linestyle='--')
        self._ax_field.set_title(f"game {self._game_id} play {self._play_id}")
        self._frame_text = self._ax_field.text(5, 51, 0, fontsize=15, color='white', ha='center')
        self._event_text = self._ax_field.text(5, 49, None, fontsize=10, color='white', ha='center')

        self.set_axis_plots(self._ax_offense, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        self.set_axis_plots(self._ax_defense, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        self.set_axis_plots(self._ax_jersey, self._MAX_FIELD_X, self._MAX_FIELD_Y)

        for idx in range(10, 120, 10):
            self._ax_field.axvline(idx, color='k', linestyle='-', alpha=0.05)

        self._ax_field.add_patch(patches.Rectangle((0, 0), 10, self._MAX_FIELD_Y,
                                                   color=self._defense_color['color2'].to_string(index=False).strip()))
        self._ax_field.add_patch(patches.Rectangle((110, 0), 10, self._MAX_FIELD_Y,
                                                   color=self._offense_color['color2'].to_string(index=False).strip()))

        self._scat_field = self._ax_field.scatter([], [], s=100, color=colors.loc[colors.team == 'FTBL']['color1'])
        self._scat_offense = self._ax_offense.scatter(
            [],
            [],
            s=500, color=self._offense_color['color1'],
            edgecolors=self._offense_color['color2'])
        self._scat_defense = self._ax_defense.scatter(
            [],
            [],
            s=500, color=self._defense_color['color1'],
            edgecolors=self._defense_color['color2'])

        self._scat_jersey_list = []
        self._scat_number_list = []
        self._scat_name_list = []
        self._vel_list = []
        self._acc_list = []
        for _ in range(self._MAX_FIELD_PLAYERS):
            self._scat_jersey_list.append(self._ax_jersey.text(
                0, 0, '', horizontalalignment='center', verticalalignment='center', c='white'))
            self._scat_number_list.append(self._ax_jersey.text(
                0, 0, '', horizontalalignment='center', verticalalignment='center', c='black'))
            self._scat_name_list.append(self._ax_jersey.text(
                0, 0, '', horizontalalignment='center', verticalalignment='center', c='black'))

            self._vel_list.append(self._ax_field.add_patch(patches.Arrow(0, 0, 0, 0, color='k')))
            self._acc_list.append(self._ax_field.add_patch(patches.Arrow(0, 0, 0, 0, color='k')))

        return (self._scat_field, self._scat_offense, self._scat_defense, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)

    def update(self, anim_frame):
        pos_df = next(self._stream)
        frameId = pos_df.frameId.unique()[0]
        event = pos_df.event.unique()[0]
        self._frame_text.set_text(str(frameId))
        self._event_text.set_text(str(event))

        for label in pos_df.team_pos.unique():
            label_data = pos_df[pos_df.team_pos == label]

            if label == 'FTBL':
                self._scat_field.set_offsets(np.hstack([label_data.x, label_data.y]))
            elif label == 'OFF':
                self._scat_offense.set_offsets(np.vstack([label_data.x, label_data.y]).T)
            elif label == 'DEF':
                self._scat_defense.set_offsets(np.vstack([label_data.x, label_data.y]).T)

        # jersey_df = pos_df[pos_df.jerseyNumber.notnull()]

        for (index, row) in pos_df[pos_df.jerseyNumber.notnull()].reset_index().iterrows():
            self._scat_jersey_list[index].set_position((row.x, row.y))
            self._scat_jersey_list[index].set_text(row.position)
            self._scat_number_list[index].set_position((row.x, row.y+1.9))
            self._scat_number_list[index].set_text(int(row.jerseyNumber))
            self._scat_name_list[index].set_position((row.x, row.y-1.9))
            self._scat_name_list[index].set_text(row.displayName.split()[-1])

            # player_orientation_rad = self.deg_to_rad(self.convert_orientation(row.o))
            # player_direction_rad = row.dir_rad
            # player_speed = row.s
            # player_accel = row.a

            # player_vel = np.array([np.real(self.polar_to_z(player_speed, player_direction_rad)),
            #                        np.imag(self.polar_to_z(player_speed, player_direction_rad))])
            # player_orient = np.array([np.real(self.polar_to_z(2, player_orientation_rad)),
            #                           np.imag(self.polar_to_z(2, player_orientation_rad))])

            self._vel_list[index].remove()
            self._vel_list[index] = self._ax_field.add_patch(
                patches.Arrow(row.x, row.y, row.v_x, row.v_y, color='k'))

            self._acc_list[index].remove()
            self._acc_list[index] = self._ax_field.add_patch(
                patches.Arrow(row.x, row.y, row.a_x, row.a_y, color='grey', width=2))

        return (self._scat_field, self._scat_offense, self._scat_defense, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)


# class AnimatePlayPitchControl(AnimatePlay):
#     def __init__(self, play_df, plot_size_len, show_control=True) -> None:
#         super().__init__(play_df, plot_size_len)
#         """Initializes the datasets used to animate the play.

#         Parameters
#         ----------
#         play_df : DataFrame
#             Dataframe corresponding to the play information for the play that requires
#             animation. This data will come from the weeks dataframe and contains position
#             and velocity information for each of the players and the football.

#         Returns
#         -------
#         None
#         """
#         self._MAX_PLAYER_SPEED = 11.3
#         self._X, self._Y, self._pos = self.generate_data_grid()

#         self._ax_football = self._ax_field.twinx()

#         self._show_control = show_control
#         plt.close()

#     @staticmethod
#     @np.vectorize
#     def radius_influence(x):
#         assert x >= 0

#         if x <= 18:
#             return 4 + (6/(18**2))*(x**2)
#         else:
#             return 10

#     def generate_data_grid(self, N=120):
#         # Our 2-dimensional distribution will be over variables X and Y
#         X = np.linspace(0, self._MAX_FIELD_X, N)
#         Y = np.linspace(0, self._MAX_FIELD_Y, N)
#         X, Y = np.meshgrid(X, Y)

#         # # Mean vector and covariance matrix
#         # mu = np.array([0., 1.])
#         # Sigma = np.array([[ 1. , -0.5], [-0.5,  1.5]])

#         # Pack X and Y into a single 3-dimensional array
#         pos = np.empty(X.shape + (2,))
#         pos[:, :, 0] = X
#         pos[:, :, 1] = Y

#         return X, Y, pos

#     @staticmethod
#     def sigmoid(x, k):
#         return 1 / (1 + np.exp(-k*x))

#     @staticmethod
#     def weighted_angle(x1, x2, w):
#         def normalize(v):
#             norm = np.linalg.norm(v, ord=1)
#             if norm == 0:
#                 norm = np.finfo(v.dtype).eps
#             return v/norm

#         norm_weighted = w*normalize(x1) + (1-w)*normalize(x2)

#         return np.arctan2(norm_weighted[1], norm_weighted[0]) % (2*np.pi)

#     @staticmethod
#     def multivariate_gaussian(pos, mu, Sigma):
#         """Return the multivariate Gaussian distribution on array pos.

#         pos is an array constructed by packing the meshed arrays of variables
#         x_1, x_2, x_3, ..., x_k into its _last_ dimension.

#         """

#         n = mu.shape[0]
#         Sigma_det = np.linalg.det(Sigma)
#         Sigma_inv = np.linalg.inv(Sigma)
#         N = np.sqrt((2*np.pi)**n * Sigma_det)
#         # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
#         # way across all the input variables.
#         fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

#         return np.exp(-fac / 2) / N

#     def generate_sigma(self, influence_rad, player_speed, distance_from_football):
#         R = np.array([[np.cos(influence_rad), -np.sin(influence_rad)], [np.sin(influence_rad), np.cos(influence_rad)]])

#         speed_ratio = (player_speed**2)/(self._MAX_PLAYER_SPEED**2)

#         S = np.array(
#             [[self.radius_influence(distance_from_football) +
#               (self.radius_influence(distance_from_football) * speed_ratio),
#               0],
#              [0, self.radius_influence(distance_from_football) -
#               (self.radius_influence(distance_from_football) * speed_ratio)]])

#         return R@(S**2)@R.T

#     def generate_mu(self, player_position, player_vel):
#         return player_position + 0.5*player_vel

#     def setup_plot(self):
#         self.set_axis_plots(self._ax_field, self._MAX_FIELD_X, self._MAX_FIELD_Y)

#         ball_snap_df = self._frame_data[(self._frame_data.event == 'ball_snap') & (self._frame_data.team == 'football')]
#         self._ax_field.axvline(ball_snap_df.x.to_numpy()[0], color='k', linestyle='--')

#         self.set_axis_plots(self._ax_offense, self._MAX_FIELD_X, self._MAX_FIELD_Y)
#         self.set_axis_plots(self._ax_defense, self._MAX_FIELD_X, self._MAX_FIELD_Y)
#         self.set_axis_plots(self._ax_jersey, self._MAX_FIELD_X, self._MAX_FIELD_Y)
#         self.set_axis_plots(self._ax_football, self._MAX_FIELD_X, self._MAX_FIELD_Y)

#         for idx in range(10, 120, 10):
#             self._ax_field.axvline(idx, color='k', linestyle='-', alpha=0.05)

#         self._scat_football = self._ax_football.scatter([], [], s=100, color='black')
#         self._scat_offense = self._ax_offense.scatter([], [], s=500, color=self._CPLT[0], edgecolors='k')
#         self._scat_defense = self._ax_defense.scatter([], [], s=500, color=self._CPLT[1], edgecolors='k')

#         self._scat_jersey_list = []
#         self._scat_number_list = []
#         self._scat_name_list = []
#         self._a_dir_list = []
#         self._a_or_list = []
#         self._inf_contours_list = []
#         for _ in range(self._MAX_FIELD_PLAYERS):
#             self._scat_jersey_list.append(self._ax_jersey.text(
#                 0, 0, '', horizontalalignment='center', verticalalignment='center', c='white'))
#             self._scat_number_list.append(self._ax_jersey.text(
#                 0, 0, '', horizontalalignment='center', verticalalignment='center', c='black'))
#             self._scat_name_list.append(self._ax_jersey.text(
#                 0, 0, '', horizontalalignment='center', verticalalignment='center', c='black'))

#             self._a_dir_list.append(self._ax_field.add_patch(patches.Arrow(0, 0, 0, 0, color='k')))
#             self._a_or_list.append(self._ax_field.add_patch(patches.Arrow(0, 0, 0, 0, color='k')))

#             if not self._show_control:
#                 self._inf_contours_list.append(self._ax_field.contourf([0, 0], [0, 0], [[0, 0], [0, 0]]))

#         if self._show_control:
#             self._pitch_control_contour = self._ax_field.contourf([0, 0], [0, 0], [[0, 0], [0, 0]])

#         return (self._scat_football, self._scat_offense, self._scat_defense, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)

#     def update(self, anim_frame):
#         pos_df = next(self._stream)

#         for label in pos_df.team_pos.unique():
#             label_data = pos_df[pos_df.team_pos == label]

#             if label == 'OFF':
#                 self._scat_offense.set_offsets(np.vstack([label_data.x, label_data.y]).T)
#             elif label == 'DEF':
#                 self._scat_defense.set_offsets(np.vstack([label_data.x, label_data.y]).T)
#             elif label == 'FTBL':
#                 self._scat_football.set_offsets(np.hstack([label_data.x, label_data.y]))

#         jersey_df = pos_df[pos_df.jerseyNumber.notnull()]

#         inf_offense_team = 0
#         inf_defense_team = 0

#         for (index, row) in pos_df[pos_df.jerseyNumber.notnull()].reset_index().iterrows():
#             self._scat_jersey_list[index].set_position((row.x, row.y))
#             self._scat_jersey_list[index].set_text(row.position)
#             self._scat_number_list[index].set_position((row.x, row.y+1.9))
#             self._scat_number_list[index].set_text(int(row.jerseyNumber))
#             self._scat_name_list[index].set_position((row.x, row.y-1.9))
#             self._scat_name_list[index].set_text(row.displayName.split()[-1])

#             player_orientation_rad = self.deg_to_rad(self.convert_orientation(row.o))
#             player_direction_rad = self.deg_to_rad(self.convert_orientation(row.dir))
#             player_speed = row.s
#             player_position = np.array([row.x, row.y])
#             player_acc = row.a

#             speed_w = player_speed/self._MAX_PLAYER_SPEED

#             player_vel = np.array([np.real(self.polar_to_z(player_speed, player_direction_rad)),
#                                    np.imag(self.polar_to_z(player_speed, player_direction_rad))])
#             player_orient = np.array([np.real(self.polar_to_z(2, player_orientation_rad)),
#                                       np.imag(self.polar_to_z(2, player_orientation_rad))])

#             influence_rad = self.weighted_angle(player_vel, player_orient, speed_w)

#             distance_from_football = np.sqrt(
#                 (pos_df[pos_df.displayName == 'Football'].x - player_position[0]) ** 2 +
#                 ((pos_df[pos_df.displayName == 'Football'].y - player_position[1])) ** 2).to_numpy()[0]

#             self._a_dir_list[index].remove()
#             self._a_dir_list[index] = self._ax_field.add_patch(
#                 patches.Arrow(row.x, row.y, player_vel[0], player_vel[1], color='k'))

#             self._a_or_list[index].remove()
#             self._a_or_list[index] = self._ax_field.add_patch(
#                 patches.Arrow(row.x, row.y, player_orient[0], player_orient[1], color='grey', width=2))

#             sigma = self.generate_sigma(influence_rad, player_speed, distance_from_football)
#             mu = self.generate_mu(player_position, player_vel)

#             Z = self.multivariate_gaussian(self._pos, mu, sigma)
#             Z_coarse = np.where(Z > 0.001, Z, np.nan)

#             if not self._show_control:
#                 for cont_info in self._inf_contours_list[index].collections:
#                     cont_info.remove()

#             if row.team_pos == 'OFF':
#                 if self._show_control:
#                     inf_offense_team += Z
#                 else:
#                     self._inf_contours_list[index] = self._ax_field.contourf(
#                         self._X, self._Y, Z_coarse, cmap='Reds', levels=10, alpha=0.1)
#             elif row.team_pos == 'DEF':
#                 if self._show_control:
#                     inf_defense_team += Z
#                 else:
#                     self._inf_contours_list[index] = self._ax_field.contourf(
#                         self._X, self._Y, Z_coarse, cmap='Greens', levels=10, alpha=0.1)

#         if self._show_control:
#             for cont_info in self._pitch_control_contour.collections:
#                 cont_info.remove()

#             self._pitch_control_contour = self._ax_field.contourf(
#                 self._X, self._Y, self.sigmoid(
#                     inf_defense_team / len(pos_df[pos_df.team_pos == 'DEF']) - inf_offense_team /
#                     len(pos_df[pos_df.team_pos == 'OFF']),
#                     k=1000),
#                 levels=50, cmap='PiYG', vmin=0.45, vmax=0.55, alpha=0.7)
#             # self._fig.colorbar(self._pitch_control_contour, extend='min', shrink=0.9, ax=self._ax_field)
#         return (self._scat_football, self._scat_offense, self._scat_defense, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)


verts = [
    (-0.4, 0.),  # left, bottom
    (-0.2, 0.2),  # left, top
    (0.2, 0.2),  # right, top
    (0.4, -0.),  # right, bottom
    (0.2, -0.2),  # left, bottom
    (-0.2, -0.2),  # left, top
    (-0.4, 0),  # left, bottom
]

codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY,
]

ballPath = Path(verts, codes, closed=True)


def create_football_field_plot(linenumbers=True,
                               endzones=True,
                               lineOfScrim=None,
                               firstDownLine=None,
                               figscale=0.1):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    """
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='darkgreen', zorder=0)

    figsize = (120*figscale, 53.3*figscale)
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    plt.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')

    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='blue',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)
    plt.xlim(0, 120)
    plt.ylim(-5, 58.3)
    plt.axis('off')
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            plt.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white')
            plt.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white', rotation=180)
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

    if lineOfScrim != None:
        hl = lineOfScrim
        plt.plot([hl, hl], [0, 53.3], color='black')
        # plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
        #          color='yellow')
    if firstDownLine != None:
        hl = lineOfScrim
        plt.plot([hl, hl], [0, 53.3], color='yellow')
        # plt.text(hl + 2, 50, '<- {}'.format(highlighted_name),
        #          color='yellow')

    return fig, ax

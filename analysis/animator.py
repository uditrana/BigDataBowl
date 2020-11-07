# Utility Libraries
from datetime import datetime
import pytz

# Computation Libraries
import numpy as np

# Plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Arrow


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
    def __init__(self, play_df, plot_size_len, normalize=False) -> None:
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

        if normalize == True:
            play_df_norm = play_df.copy(deep=True)
            play_df_norm = play_df_norm.drop(['x', 'y', 'o', 'dir'], axis=1).rename(
                columns={'x_norm': 'x', 'y_norm': 'y', 'o_norm': 'o', 'dir_norm': 'dir'})
            play_df = play_df_norm

        self._CPLT = sns.color_palette("husl", 2)
        self._frame_data = play_df
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

        self._ax_home = self._ax_field.twinx()
        self._ax_away = self._ax_field.twinx()
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

        ball_snap_df = self._frame_data[(self._frame_data.event == 'ball_snap') & (self._frame_data.team == 'football')]
        self._ax_field.axvline(ball_snap_df.x.to_numpy()[0], color='k', linestyle='--')

        self.set_axis_plots(self._ax_home, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        self.set_axis_plots(self._ax_away, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        self.set_axis_plots(self._ax_jersey, self._MAX_FIELD_X, self._MAX_FIELD_Y)

        for idx in range(10, 120, 10):
            self._ax_field.axvline(idx, color='k', linestyle='-', alpha=0.05)

        self._scat_field = self._ax_field.scatter([], [], s=100, color='black')
        self._scat_home = self._ax_home.scatter([], [], s=500, color=self._CPLT[0], edgecolors='k')
        self._scat_away = self._ax_away.scatter([], [], s=500, color=self._CPLT[1], edgecolors='k')

        self._scat_jersey_list = []
        self._scat_number_list = []
        self._scat_name_list = []
        self._a_dir_list = []
        self._a_or_list = []
        for _ in range(self._MAX_FIELD_PLAYERS):
            self._scat_jersey_list.append(self._ax_jersey.text(
                0, 0, '', horizontalalignment='center', verticalalignment='center', c='white'))
            self._scat_number_list.append(self._ax_jersey.text(
                0, 0, '', horizontalalignment='center', verticalalignment='center', c='black'))
            self._scat_name_list.append(self._ax_jersey.text(
                0, 0, '', horizontalalignment='center', verticalalignment='center', c='black'))

            self._a_dir_list.append(self._ax_field.add_patch(Arrow(0, 0, 0, 0, color='k')))
            self._a_or_list.append(self._ax_field.add_patch(Arrow(0, 0, 0, 0, color='k')))

        return (self._scat_field, self._scat_home, self._scat_away, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)

    def update(self, anim_frame):
        pos_df = next(self._stream)

        for label in pos_df.team.unique():
            label_data = pos_df[pos_df.team == label]

            if label == 'football':
                self._scat_field.set_offsets(np.hstack([label_data.x, label_data.y]))
            elif label == 'home':
                self._scat_home.set_offsets(np.vstack([label_data.x, label_data.y]).T)
            elif label == 'away':
                self._scat_away.set_offsets(np.vstack([label_data.x, label_data.y]).T)

        jersey_df = pos_df[pos_df.jerseyNumber.notnull()]

        for (index, row) in pos_df[pos_df.jerseyNumber.notnull()].reset_index().iterrows():
            self._scat_jersey_list[index].set_position((row.x, row.y))
            self._scat_jersey_list[index].set_text(row.position)
            self._scat_number_list[index].set_position((row.x, row.y+1.9))
            self._scat_number_list[index].set_text(int(row.jerseyNumber))
            self._scat_name_list[index].set_position((row.x, row.y-1.9))
            self._scat_name_list[index].set_text(row.displayName.split()[-1])

            player_orientation_rad = self.deg_to_rad(self.convert_orientation(row.o))
            player_direction_rad = self.deg_to_rad(self.convert_orientation(row.dir))
            player_speed = row.s

            player_vel = np.array([np.real(self.polar_to_z(player_speed, player_direction_rad)),
                                   np.imag(self.polar_to_z(player_speed, player_direction_rad))])
            player_orient = np.array([np.real(self.polar_to_z(2, player_orientation_rad)),
                                      np.imag(self.polar_to_z(2, player_orientation_rad))])

            self._a_dir_list[index].remove()
            self._a_dir_list[index] = self._ax_field.add_patch(
                Arrow(row.x, row.y, player_vel[0], player_vel[1], color='k'))

            self._a_or_list[index].remove()
            self._a_or_list[index] = self._ax_field.add_patch(
                Arrow(row.x, row.y, player_orient[0], player_orient[1], color='grey', width=2))

        return (self._scat_field, self._scat_home, self._scat_away, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)


class AnimatePlayPitchControl(AnimatePlay):
    def __init__(self, play_df, plot_size_len, show_control=True) -> None:
        super().__init__(play_df, plot_size_len)
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
        self._MAX_PLAYER_SPEED = 11.3
        self._X, self._Y, self._pos = self.generate_data_grid()

        self._ax_football = self._ax_field.twinx()

        self._show_control = show_control
        plt.close()

    @staticmethod
    @np.vectorize
    def radius_influence(x):
        assert x >= 0

        if x <= 18:
            return 4 + (6/(18**2))*(x**2)
        else:
            return 10

    def generate_data_grid(self, N=120):
        # Our 2-dimensional distribution will be over variables X and Y
        X = np.linspace(0, self._MAX_FIELD_X, N)
        Y = np.linspace(0, self._MAX_FIELD_Y, N)
        X, Y = np.meshgrid(X, Y)

        # # Mean vector and covariance matrix
        # mu = np.array([0., 1.])
        # Sigma = np.array([[ 1. , -0.5], [-0.5,  1.5]])

        # Pack X and Y into a single 3-dimensional array
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X
        pos[:, :, 1] = Y

        return X, Y, pos

    @staticmethod
    def sigmoid(x, k):
        return 1 / (1 + np.exp(-k*x))

    @staticmethod
    def weighted_angle(x1, x2, w):
        def normalize(v):
            norm = np.linalg.norm(v, ord=1)
            if norm == 0:
                norm = np.finfo(v.dtype).eps
            return v/norm

        norm_weighted = w*normalize(x1) + (1-w)*normalize(x2)

        return np.arctan2(norm_weighted[1], norm_weighted[0]) % (2*np.pi)

    @staticmethod
    def multivariate_gaussian(pos, mu, Sigma):
        """Return the multivariate Gaussian distribution on array pos.

        pos is an array constructed by packing the meshed arrays of variables
        x_1, x_2, x_3, ..., x_k into its _last_ dimension.

        """

        n = mu.shape[0]
        Sigma_det = np.linalg.det(Sigma)
        Sigma_inv = np.linalg.inv(Sigma)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

        return np.exp(-fac / 2) / N

    def generate_sigma(self, influence_rad, player_speed, distance_from_football):
        R = np.array([[np.cos(influence_rad), -np.sin(influence_rad)], [np.sin(influence_rad), np.cos(influence_rad)]])

        speed_ratio = (player_speed**2)/(self._MAX_PLAYER_SPEED**2)

        S = np.array(
            [[self.radius_influence(distance_from_football) +
              (self.radius_influence(distance_from_football) * speed_ratio),
              0],
             [0, self.radius_influence(distance_from_football) -
              (self.radius_influence(distance_from_football) * speed_ratio)]])

        return R@(S**2)@R.T

    def generate_mu(self, player_position, player_vel):
        return player_position + 0.5*player_vel

    def setup_plot(self):
        self.set_axis_plots(self._ax_field, self._MAX_FIELD_X, self._MAX_FIELD_Y)

        ball_snap_df = self._frame_data[(self._frame_data.event == 'ball_snap') & (self._frame_data.team == 'football')]
        self._ax_field.axvline(ball_snap_df.x.to_numpy()[0], color='k', linestyle='--')

        self.set_axis_plots(self._ax_home, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        self.set_axis_plots(self._ax_away, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        self.set_axis_plots(self._ax_jersey, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        self.set_axis_plots(self._ax_football, self._MAX_FIELD_X, self._MAX_FIELD_Y)

        for idx in range(10, 120, 10):
            self._ax_field.axvline(idx, color='k', linestyle='-', alpha=0.05)

        self._scat_football = self._ax_football.scatter([], [], s=100, color='black')
        self._scat_home = self._ax_home.scatter([], [], s=500, color=self._CPLT[0], edgecolors='k')
        self._scat_away = self._ax_away.scatter([], [], s=500, color=self._CPLT[1], edgecolors='k')

        self._scat_jersey_list = []
        self._scat_number_list = []
        self._scat_name_list = []
        self._a_dir_list = []
        self._a_or_list = []
        self._inf_contours_list = []
        for _ in range(self._MAX_FIELD_PLAYERS):
            self._scat_jersey_list.append(self._ax_jersey.text(
                0, 0, '', horizontalalignment='center', verticalalignment='center', c='white'))
            self._scat_number_list.append(self._ax_jersey.text(
                0, 0, '', horizontalalignment='center', verticalalignment='center', c='black'))
            self._scat_name_list.append(self._ax_jersey.text(
                0, 0, '', horizontalalignment='center', verticalalignment='center', c='black'))

            self._a_dir_list.append(self._ax_field.add_patch(Arrow(0, 0, 0, 0, color='k')))
            self._a_or_list.append(self._ax_field.add_patch(Arrow(0, 0, 0, 0, color='k')))

            if not self._show_control:
                self._inf_contours_list.append(self._ax_field.contourf([0, 0], [0, 0], [[0, 0], [0, 0]]))

        if self._show_control:
            self._pitch_control_contour = self._ax_field.contourf([0, 0], [0, 0], [[0, 0], [0, 0]])

        return (self._scat_football, self._scat_home, self._scat_away, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)

    def update(self, anim_frame):
        pos_df = next(self._stream)

        for label in pos_df.team.unique():
            label_data = pos_df[pos_df.team == label]

            if label == 'home':
                self._scat_home.set_offsets(np.vstack([label_data.x, label_data.y]).T)
            elif label == 'away':
                self._scat_away.set_offsets(np.vstack([label_data.x, label_data.y]).T)
            elif label == 'football':
                self._scat_football.set_offsets(np.hstack([label_data.x, label_data.y]))

        jersey_df = pos_df[pos_df.jerseyNumber.notnull()]

        inf_home_team = 0
        inf_away_team = 0

        for (index, row) in pos_df[pos_df.jerseyNumber.notnull()].reset_index().iterrows():
            self._scat_jersey_list[index].set_position((row.x, row.y))
            self._scat_jersey_list[index].set_text(row.position)
            self._scat_number_list[index].set_position((row.x, row.y+1.9))
            self._scat_number_list[index].set_text(int(row.jerseyNumber))
            self._scat_name_list[index].set_position((row.x, row.y-1.9))
            self._scat_name_list[index].set_text(row.displayName.split()[-1])

            player_orientation_rad = self.deg_to_rad(self.convert_orientation(row.o))
            player_direction_rad = self.deg_to_rad(self.convert_orientation(row.dir))
            player_speed = row.s
            player_position = np.array([row.x, row.y])
            player_acc = row.a

            speed_w = player_speed/self._MAX_PLAYER_SPEED

            player_vel = np.array([np.real(self.polar_to_z(player_speed, player_direction_rad)),
                                   np.imag(self.polar_to_z(player_speed, player_direction_rad))])
            player_orient = np.array([np.real(self.polar_to_z(2, player_orientation_rad)),
                                      np.imag(self.polar_to_z(2, player_orientation_rad))])

            influence_rad = self.weighted_angle(player_vel, player_orient, speed_w)

            distance_from_football = np.sqrt(
                (pos_df[pos_df.displayName == 'Football'].x - player_position[0]) ** 2 +
                ((pos_df[pos_df.displayName == 'Football'].y - player_position[1])) ** 2).to_numpy()[0]

            self._a_dir_list[index].remove()
            self._a_dir_list[index] = self._ax_field.add_patch(
                Arrow(row.x, row.y, player_vel[0], player_vel[1], color='k'))

            self._a_or_list[index].remove()
            self._a_or_list[index] = self._ax_field.add_patch(
                Arrow(row.x, row.y, player_orient[0], player_orient[1], color='grey', width=2))

            sigma = self.generate_sigma(influence_rad, player_speed, distance_from_football)
            mu = self.generate_mu(player_position, player_vel)

            Z = self.multivariate_gaussian(self._pos, mu, sigma)
            Z_coarse = np.where(Z > 0.001, Z, np.nan)

            if not self._show_control:
                for cont_info in self._inf_contours_list[index].collections:
                    cont_info.remove()

            if row.team == 'home':
                if self._show_control:
                    inf_home_team += Z
                else:
                    self._inf_contours_list[index] = self._ax_field.contourf(
                        self._X, self._Y, Z_coarse, cmap='Reds', levels=10, alpha=0.1)
            elif row.team == 'away':
                if self._show_control:
                    inf_away_team += Z
                else:
                    self._inf_contours_list[index] = self._ax_field.contourf(
                        self._X, self._Y, Z_coarse, cmap='Greens', levels=10, alpha=0.1)

        if self._show_control:
            for cont_info in self._pitch_control_contour.collections:
                cont_info.remove()

            self._pitch_control_contour = self._ax_field.contourf(
                self._X, self._Y, self.sigmoid(
                    inf_away_team / len(pos_df[pos_df.team == 'away']) - inf_home_team /
                    len(pos_df[pos_df.team == 'home']),
                    k=1000),
                levels=50, cmap='PiYG', vmin=0.45, vmax=0.55, alpha=0.7)
            # self._fig.colorbar(self._pitch_control_contour, extend='min', shrink=0.9, ax=self._ax_field)
        return (self._scat_football, self._scat_home, self._scat_away, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)

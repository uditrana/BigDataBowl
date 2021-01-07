# Utility Libraries
from datetime import datetime
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from pandas.core import frame
import pytz
import pandas as pd

# Computation Libraries
import numpy as np

# Plotting libraries
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from matplotlib.path import Path

from mpl_toolkits.axes_grid1 import make_axes_locatable

# colors
# import seaborn as sns

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


colors = pd.read_csv(
    'https://raw.githubusercontent.com/uditrana/NFLFastR_analysis/master/nfl_colors.tsv', delimiter='\t')


class AnimatePlay:
    def __init__(self, play_df, plot_size_len, field_prob_df=None, viz_proj=False) -> None:
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
        self.YARD_PIXEL_COUNT = 70

        self._show_p_mass = type(field_prob_df) != type(None) and ('p_mass' in field_prob_df.columns)
        self._show_v_mass = type(field_prob_df) != type(None) and ('v_mass' in field_prob_df.columns)
        self._viz_proj = viz_proj
        self._field_prob_df = field_prob_df
        # self._CPLT = sns.color_palette("husl", 2)
        self._offense_color = colors.loc[colors.team == play_df.loc[play_df.team_pos == 'OFF']['teamAbbr'].iloc[0]]
        # self._offense_color = colors.loc[colors.team == 'NFC']
        self._defense_color = colors.loc[colors.team == play_df.loc[play_df.team_pos == 'DEF']['teamAbbr'].iloc[0]]
        # self._defense_color = colors.loc[colors.team == 'AFC']
        self._offense_colors = [
            self._offense_color['color1'].to_string(index=False).strip(),
            self._offense_color['color2'].to_string(index=False).strip()]
        self._defense_colors = [
            self._defense_color['color1'].to_string(index=False).strip(),
            self._defense_color['color2'].to_string(index=False).strip()]
        if self._offense_color['color1_family'].to_string(
                index=False).strip() == self._defense_color['color1_family'].to_string(
                index=False).strip():
            self._defense_colors = self._defense_colors[::-1]

        # try:
        #     self._pass_arrival_loc = play_df.loc[(play_df.event == 'pass_arrived') &
        #                                          (play_df.nflId == 0)][['x', 'y']].iloc[0].to_numpy()
        # except:
        #     self._pass_arrival_loc = np.array([-10, -10])
        # print(self._pass_arrival_loc, type(self._pass_arrival_loc))

        # print(self._offense_color, self._defense_color, self._offense_colors, self._defense_colors)

        self._frame_data = play_df
        self._game_id, self._play_id = play_df.iloc[:1][['gameId', 'playId']].to_records(index=False)[0]
        self._times = sorted(play_df.time.unique())
        self._frames = sorted(play_df.frameId.unique())
        self._stream = self.data_stream()

        self._date_format = "%Y-%m-%dT%H:%M:%S.%fZ"
        self._mean_interval_ms = np.mean(
            [delta.microseconds / 1000
             for delta in np.diff(
                 np.array(
                     [pytz.timezone('US/Eastern').localize(datetime.strptime(date_string, self._date_format))
                      for date_string in self._times]))])
        self._mean_interval_ms = 100

        self._fig = plt.figure(figsize=(plot_size_len, plot_size_len*(self._MAX_FIELD_Y/self._MAX_FIELD_X)))

        self._ax_field = plt.gca()

        # divider = make_axes_locatable(self._ax_field)
        # self._cax = divider.append_axes('right', size='2%', pad=0.1)
        # self._cax = self._fig.add_axes([0.91, 0.13, 0.02, 0.75])

        self._ax_offense = self._ax_field.twinx()
        self._ax_defense = self._ax_field.twinx()
        self._ax_jersey = self._ax_field.twinx()

        print(len(self._frames))
        self.ani = animation.FuncAnimation(
            self._fig, self.update, frames=len(self._frames),
            interval=self._mean_interval_ms, init_func=self.setup_plot, blit=False)

        plt.close()

    @ staticmethod
    def set_axis_plots(ax, max_x, max_y) -> None:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

        ax.set_xlim([0, max_x])
        ax.set_ylim([0, max_y])

    @ staticmethod
    def convert_orientation(x):
        return (-x + 90) % 360

    @ staticmethod
    def polar_to_z(r, theta):
        return r * np.exp(1j * theta)

    @ staticmethod
    def deg_to_rad(deg):
        return deg*np.pi/180

    def data_stream(self):
        for frame in self._frames:
            yield self._frame_data[self._frame_data.frameId == frame]

    def setup_plot(self):
        self.set_axis_plots(self._ax_field, self._MAX_FIELD_X, self._MAX_FIELD_Y)

        # ball_snap_df = self._frame_data[(self._frame_data.event == 'ball_snap') & (self._frame_data.team == 'football')]
        self._ax_field.axvline(self._frame_data.iloc[0]['los'], color='k', linestyle='--')
        # self._ax_field.set_title(f"game {self._game_id} play {self._play_id}", c='black')
        self._frame_text = self._ax_field.text(5, 51, 0, fontsize=15, color='black', ha='center')
        self._event_text = self._ax_field.text(5, 49, None, fontsize=8, color='black', ha='center')
        # self._ball_loc_text = self._ax_field.text(5, 47, None, fontsize=12, color='black', ha='center')
        # self._pass_arr_text = self._ax_field.text(5, 45, None, fontsize=12, color='black', ha='center')

        self.set_axis_plots(self._ax_offense, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        self.set_axis_plots(self._ax_defense, self._MAX_FIELD_X, self._MAX_FIELD_Y)
        self.set_axis_plots(self._ax_jersey, self._MAX_FIELD_X, self._MAX_FIELD_Y)

        for idx in range(10, 120, 10):
            self._ax_field.axvline(idx, color='k', linestyle='-', alpha=0.05)
            self._ax_field.text(idx+.1, 2, str(idx-10), rotation=90, color='k', alpha=0.2)

        self._ax_field.add_patch(patches.Rectangle((0, 0), 10, self._MAX_FIELD_Y,
                                                   color='black', alpha=0.1))
        self._ax_field.add_patch(patches.Rectangle((110, 0), 10, self._MAX_FIELD_Y,
                                                   color='black', alpha=0.1))

        if self._show_p_mass:
            pmin, pmax = 0, 1
            norm = mpl.colors.Normalize(vmin=pmin, vmax=pmax)
            col = self._offense_colors[1] if self._offense_color['color2_family'].to_string(
                index=False).strip() != 'black' else self._offense_colors[0]
            prob_cm = LinearSegmentedColormap.from_list('probability', [(0, 'white'), (1, col)])
            self._scat_field_pmass = self._ax_field.scatter(
                [],
                [],
                s=self.YARD_PIXEL_COUNT, marker='s', alpha=0.7, cmap=prob_cm, vmin=pmin, vmax=pmax)
            self._scat_field_pmass.set_cmap(prob_cm)
            self._fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=prob_cm), ax=self._ax_field, orientation='vertical')

        if self._show_v_mass:
            vmin, vmax = self._field_prob_df.v_mass.min(), self._field_prob_df.v_mass.max()
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            col = self._offense_colors[1] if self._offense_color['color2_family'].to_string(
                index=False).strip() not in ['black', 'gray'] else self._offense_colors[0]
            if vmin >= 0:
                val_cm = LinearSegmentedColormap.from_list('value', [(0, 'white'), (1, col)])
            else:
                val_cm = LinearSegmentedColormap.from_list(
                    'value', [(0, 'black'), (abs(vmin / (vmax - vmin)) if vmax * vmin < 0 else 0, 'white'), (1, col)])
            self._scat_field_vmass = self._ax_field.scatter(
                [],
                [],
                s=self.YARD_PIXEL_COUNT, marker='s', c=[], cmap=val_cm, vmin=vmin, vmax=vmax, alpha=0.7)
            self._scat_field_vmass.set_cmap(val_cm)
            self._fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=val_cm), ax=self._ax_field, orientation='vertical')

        self._scat_offense = self._ax_offense.scatter(
            [],
            [],
            s=300, color=self._offense_colors[0],
            edgecolors=self._offense_colors[1])
        self._scat_offense_proj = self._ax_offense.scatter(
            [],
            [],
            s=100, color=self._offense_colors[0], alpha=0.6, marker='x',
            edgecolors=self._offense_colors[1])
        # self._scat_offense_reax = self._ax_offense.scatter(
        #     [],
        #     [],
        #     s=100, color=self._offense_colors[0], alpha=0.5, marker='*',
        #     edgecolors=self._offense_colors[1])
        self._scat_defense = self._ax_defense.scatter(
            [],
            [],
            s=300, color=self._defense_colors[0],
            edgecolors=self._defense_colors[1])
        self._scat_defense_proj = self._ax_defense.scatter(
            [],
            [],
            s=100, color=self._defense_colors[0], alpha=0.5, marker='x',
            edgecolors=self._defense_colors[1])
        # self._scat_defense_reax = self._ax_defense.scatter(
        #     [],
        #     [],
        #     s=100, color=self._defense_colors[0], alpha=0.5, marker='*',
        #     edgecolors=self._defense_colors[1])

        self._scat_jersey_list = []
        self._scat_number_list = []
        self._scat_number_proj_list = []
        # self._scat_number_reax_list = []
        self._scat_name_list = []
        self._vel_list = []
        self._acc_list = []
        self._vel_proj_list = []
        self._acc_proj_list = []
        for _ in range(self._MAX_FIELD_PLAYERS):
            self._scat_jersey_list.append(self._ax_jersey.text(
                0, 0, '', horizontalalignment='center', verticalalignment='center', c='white', size=8))
            self._scat_number_list.append(self._ax_jersey.text(
                0, 0, '', horizontalalignment='center', verticalalignment='center', c='black'))
            self._scat_number_proj_list.append(self._ax_jersey.text(
                0, 0, '', horizontalalignment='center', verticalalignment='center', c='black'))
            # self._scat_number_reax_list.append(self._ax_jersey.text(
            #     0, 0, '', horizontalalignment='center', verticalalignment='center', c='black'))
            self._scat_name_list.append(self._ax_jersey.text(
                0, 0, '', horizontalalignment='center', verticalalignment='center', c='black'))

            self._vel_list.append(self._ax_field.add_patch(patches.Arrow(0, 0, 0, 0, color='k')))
            self._acc_list.append(self._ax_field.add_patch(patches.Arrow(0, 0, 0, 0, color='k')))
            self._vel_proj_list.append(self._ax_field.add_patch(patches.Arrow(0, 0, 0, 0, color='k')))
            self._acc_proj_list.append(self._ax_field.add_patch(patches.Arrow(0, 0, 0, 0, color='k')))
            # self._or_list.append(self._ax_field.add_patch(patches.Arrow(0, 0, 0, 0, color='k')))

        self._scat_field = self._ax_field.scatter(
            [], [], s=200, color=colors.loc[colors.team == 'FTBL']['color1'], marker=ballPath)

        return (self._scat_field, self._scat_offense, self._scat_defense, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)

    def update(self, anim_frame):
        try:
            pos_df = next(self._stream)
        except StopIteration:
            return (self._scat_field, self._scat_offense, self._scat_defense, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)

        frameId = pos_df.frameId.unique()[0]
        event = pos_df.event.unique()[0]
        self._frame_text.set_text(str(frameId))
        self._event_text.set_text(str(event))
        # self._ball_loc_text.set_text(str(pos_df.loc[pos_df.nflId == 0][['x', 'y']].to_records(index=False)))
        # self._pass_arr_text.set_text(str(self._pass_arrival_loc))

        for label in pos_df.team_pos.unique():
            label_data = pos_df[pos_df.team_pos == label]

            if label == 'FTBL':
                # self._scat_field.set_offsets(np.hstack([label_data.x, label_data.y]))
                self._scat_field.set_offsets(np.vstack([label_data.x, label_data.y]).T)
            elif label == 'OFF':
                self._scat_offense.set_offsets(np.vstack([label_data.x, label_data.y]).T)
                if self._viz_proj:
                    self._scat_offense_proj.set_offsets(
                        np.vstack([label_data.proj_x.fillna(-10), label_data.proj_y.fillna(-10)]).T)
                    # self._scat_offense_reax.set_offsets(
                    #     np.vstack([label_data.reax_x.fillna(-10), label_data.reax_y.fillna(-10)]).T)
            elif label == 'DEF':
                self._scat_defense.set_offsets(np.vstack([label_data.x, label_data.y]).T)
                if self._viz_proj:
                    self._scat_defense_proj.set_offsets(
                        np.vstack([label_data.proj_x.fillna(-10), label_data.proj_y.fillna(-10)]).T)
                    # self._scat_defense_reax.set_offsets(
                    #     np.vstack([label_data.reax_x.fillna(-10), label_data.reax_y.fillna(-10)]).T)

        # jersey_df = pos_df[pos_df.jerseyNumber.notnull()]

        if self._show_p_mass:
            frame_prob_df = self._field_prob_df.loc[self._field_prob_df.frameId == frameId]
            # print(f"frameId: {frameId}, ")
            if len(frame_prob_df > 0):
                try:
                    self._scat_field_pmass.set_array(frame_prob_df['p_mass'].to_numpy())
                    self._scat_field_pmass.set_offsets(
                        np.vstack([frame_prob_df.ball_end_x, frame_prob_df.ball_end_y]).T)
                    # self._scat_control.set_cmap(mpl.colors.Colormap.ListedColormap(['red', 'white', 'blue']))
                    # self._scat_field_pmass1.set_cmap('bwr')
                except:
                    pass

        if self._show_v_mass:
            frame_prob_df = self._field_prob_df.loc[self._field_prob_df.frameId == frameId]
            # print(f"frameId: {frameId}, ")
            if len(frame_prob_df > 0):
                try:
                    self._scat_field_vmass.set_array(frame_prob_df['v_mass'].to_numpy())
                    # grayscale = 50
                    # self._scat_field_vmass.set_color([(grayscale/255, grayscale/255, grayscale/255, p)
                    #                                   for p in np.clip(frame_prob_df['v_mass'], 0, 1)])
                    self._scat_field_vmass.set_offsets(
                        np.vstack([frame_prob_df.ball_end_x, frame_prob_df.ball_end_y]).T)
                except:
                    pass

            # else:
            #     self._scat_field_pmass1.set_offsets()
            #     self._scat_field_vmass.set_offsets()

        for (index, row) in pos_df[pos_df.jerseyNumber.notnull()].reset_index().iterrows():
            row = row.fillna(-10)
            self._scat_jersey_list[index].set_position((row.x, row.y))
            self._scat_jersey_list[index].set_text(row.position)
            self._scat_number_list[index].set_position((row.x, row.y+1.5))
            self._scat_number_list[index].set_text(int(row.jerseyNumber))
            if self._viz_proj:
                self._scat_number_proj_list[index].set_position((row.proj_x, row.proj_y+1.5))
                self._scat_number_proj_list[index].set_text(int(row.jerseyNumber))

                try:
                    self._vel_proj_list[index].remove()
                    self._vel_proj_list[index] = self._ax_field.add_patch(
                        patches.Arrow(row.proj_x, row.proj_y, row.proj_v_x, row.proj_v_y, color='k', width=0.5))
                except:
                    pass

                try:
                    self._acc_proj_list[index].remove()
                    self._acc_proj_list[index] = self._ax_field.add_patch(
                        patches.Arrow(row.proj_x, row.proj_y, row.proj_a_x, row.proj_a_y, color='grey', width=0.75))
                except:
                    pass

            self._scat_name_list[index].set_position((row.x, row.y-1.5))
            self._scat_name_list[index].set_text(row.displayName.split()[-1])

            # player_orientation_rad = self.deg_to_rad(self.convert_orientation(row.o))
            # player_orient = np.array([np.real(self.polar_to_z(2, player_orientation_rad)),
            #                           np.imag(self.polar_to_z(2, player_orientation_rad))])

            self._vel_list[index].remove()
            self._vel_list[index] = self._ax_field.add_patch(
                patches.Arrow(row.x, row.y, row.v_x, row.v_y, color='k', width=0.75))

            self._acc_list[index].remove()
            self._acc_list[index] = self._ax_field.add_patch(
                patches.Arrow(row.x, row.y, row.a_x, row.a_y, color='grey', width=1))

            # self._or_list[index].remove()
            # self._or_list[index] = self._ax_field.add_patch(
            #     patches.Arrow(row.x, row.y, row.a_x, row.a_y, color='grey', width=2))

        return (self._scat_field, self._scat_offense, self._scat_defense, *self._scat_jersey_list, *self._scat_number_list, *self._scat_name_list)


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

import os
import csv
from environments.climb_game import ClimbGame
from environments.RO3 import R03Game
from environments.extended_climb_game import ExtendedClimbGame
from environments.extended_st_climb_game import ExtendedStochasticClimbGame
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def get_lenient_parameters(game, game_type):
    """
    Get the leniency moderation factor, minimum and maximum reward for a specific game and game type
    :param game: Climb Game, Extended (stochastic) Climb or Relative Overgeneralization
    :param game_type: det, ps or fs
    :return:
    """
    if isinstance(game, ClimbGame) or isinstance(game, ExtendedClimbGame):
        omega = 1
        if game_type == 'det':
            leniency = 10 ** 7
            min_r = -30
            max_r = 11
        elif game_type == 'ps':
            leniency = 10 ** 3
            min_r = -30
            max_r = 14
        elif game_type == 'fs':
            leniency = 10
            min_r = -65
            max_r = 14
        elif game_type == 'normal':
            scale = 5
            leniency = 10
            min_r = -30 - (scale * 3)
            max_r = 7 + (scale * 3)
        else:
            print('put valid game type: det, ps or fs')
            min_r, max_r, leniency, omega = None, None, None, None
    elif isinstance(game, R03Game):
        leniency = 1
        omega = 0.3
        min_r = 0
        max_r = 10
    elif isinstance(game, ExtendedStochasticClimbGame):
        omega = 1
        leniency = 10 ** 3
        min_r = -30
        max_r = 14
    else:
        print('put valid game (Climb Game, Extended Climb Game or Relative Overgeneralization 3)')
        min_r, max_r, leniency, omega = None, None, None, None

    return min_r, max_r, leniency, omega


def plot_ellipses():
    """Plot the three ellipses used in figure 17 in my thesis"""
    ellipse1 = Ellipse((9820, 8.97), width=500, height=4.5, zorder=10, fill=False, color='black')
    ellipse2 = Ellipse((12620, 8.98), width=500, height=4.5, zorder=10, fill=False, color='black')
    ellipse3 = Ellipse((13070, 5.995), width=500, height=4.5, zorder=10, fill=False, color='black')
    plt.gcf().gca().add_artist(ellipse1)
    plt.gcf().gca().add_artist(ellipse2)
    plt.gcf().gca().add_artist(ellipse3)


def write_to_csv(e_decays, algo_name, metric, num_episodes, correct_policy_results, game_type, beta,
                 sample_efficiency_mean_results, sample_efficiency_std_results, sample_efficiency_list_results, custom):
    """Make directory and store 4 csv files. The correct policies, the mean sample efficiency,
       the std sample efficiency and the list of all sample efficiencies per run organized in a grid with the epsilon
       decay as rows and the temperature decay as columns.
    """
    folder = f'data_{algo_name}_{game_type}_{metric}_b{beta}_{num_episodes}_{custom}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    fields = ['Temp decay'] + e_decays
    filename = f'cp_{algo_name}_{game_type}_{metric}_b{beta}_{num_episodes}.csv'
    path = os.path.join(folder, filename)
    with open(path, 'w') as csvfile_cp:
        csvwriter_cp = csv.writer(csvfile_cp)
        csvwriter_cp.writerow(fields)
        csvwriter_cp.writerows(correct_policy_results)
    filename = f'se_mean_{algo_name}_{game_type}_{metric}_b{beta}_{num_episodes}.csv'
    path = os.path.join(folder, filename)
    with open(path, 'w') as csvfile_se_mean:
        csvwriter_se_mean = csv.writer(csvfile_se_mean)
        csvwriter_se_mean.writerow(fields)
        csvwriter_se_mean.writerows(sample_efficiency_mean_results)
    filename = f'se_std_{algo_name}_{game_type}_{metric}_b{beta}_{num_episodes}.csv'
    path = os.path.join(folder, filename)
    with open(path, 'w') as csvfile_se_std:
        csvwriter_se_std = csv.writer(csvfile_se_std)
        csvwriter_se_std.writerow(fields)
        csvwriter_se_std.writerows(sample_efficiency_std_results)
    filename = f'se_list_{algo_name}_{game_type}_{metric}_b{beta}_{num_episodes}.csv'
    path = os.path.join(folder, filename)
    with open(path, 'w') as csvfile_se_list:
        csvwriter_se_list = csv.writer(csvfile_se_list)
        csvwriter_se_list.writerows(sample_efficiency_list_results)

import os
import csv
from environments.climb_game import ClimbGame
from environments.RO3 import R03Game
from environments.extended_climb_game import ExtendedClimbGame
from environments.extended_st_climb_game import ExtendedStochasticClimbGame
from environments.extended_double_climb_game import ExtendedDoubleClimbGame
from environments.extended_RO3_climb_game import ExtendedROClimbGame
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def get_lenient_parameters(game, game_type):
    if isinstance(game, ClimbGame) or isinstance(game, ExtendedClimbGame):
        omega = 1
        if game_type == 'det':
            leniency = 10 ** 7
            min_r = -30
            max_r = 11
        elif game_type == 'ps' or game_type == 'ps2':
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
    elif isinstance(game, ExtendedDoubleClimbGame):
        omega = 1
        leniency = 10 ** 3
        min_r = -41
        max_r = 14
    elif isinstance(game, ExtendedROClimbGame):
        omega = 1
        leniency = 10 ** 3
        min_r = -71
        max_r = 14
    else:
        print('put valid game (Climbgame, ExtendedClimbGame or RO3)')
        min_r, max_r, leniency, omega = None, None, None, None

    return min_r, max_r, leniency, omega


def plot_ellipses():
    ellipse1 = Ellipse((9820, 8.97), width=500, height=4.5, zorder=10, fill=False, color='black')
    ellipse2 = Ellipse((12620, 8.98), width=500, height=4.5, zorder=10, fill=False, color='black')
    ellipse3 = Ellipse((13070, 5.995), width=500, height=4.5, zorder=10, fill=False, color='black')
    plt.gcf().gca().add_artist(ellipse1)
    plt.gcf().gca().add_artist(ellipse2)
    plt.gcf().gca().add_artist(ellipse3)


def write_to_csv(e_decays, algo_name, metric, num_episodes, correct_policy_results, game_type, beta,
                 sample_efficiency_mean_results, sample_efficiency_std_results, sample_efficiency_list_results, custom):
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

# Overige rotzooi ---------------------------------------------------------------------------

# bins = np.linspace(0, 10 + 0.000001, 10)
# samples = [-1, 1, 2, 3, 4, 5, 6, 7, 11]
# cur_hist, _ = np.histogram(samples, bins)
# print(bins)
# print(cur_hist)

# samples = [[10, 20, 50, 100, 200, 500, 1000]]
# e_mean = [[0.23, 0.25, 0.30, 0.39, 0.54, 1.02, 2.2]]
# k_mean = [[0.27, 0.31, 0.44, 0.67, 1.15, 2.52, 4.3]]
# h_mean = [[0.92, 0.93, 1.05, 1.1, 1.36, 1.90, 2.6]]
# js_mean = [[1.1, 1.24, 1.27, 1.31, 1.51, 2.05, 2.66]]
# tdl_mean = [[0.47, 0.53, 0.55, 0.60, 0.88, 1.39, 2.0]]
# o_mean = [[0.81, 0.81, 0.86, 0.96, 1.22, 1.74, 2.35]]
#
# e_std = [[0.005, 0.005, 0.007, 0.01, 0.025, 0.05, 0.1]]
# k_std = [[0.005, 0.01, 0.01, 0.04, 0.05, 0.12, 0.05]]
# h_std = [[0.04, 0.01, 0.02, 0.02, 0.06, 0.1, 0.3]]
# js_std = [[0.01, 0.02, 0.02, 0.04, 0.07, 0.1, 0.04]]
# tdl_std = [[0.01, 0.01, 0.01, 0.01, 0.04, 0.07, 0.1]]
# o_std = [[0.02, 0.01, 0.01, 0.02, 0.06, 0.08, 0.1]]
# #
# tot_mean = np.concatenate((k_mean, e_mean, js_mean, h_mean, o_mean, tdl_mean))
# tot_std = np.concatenate((k_std, e_std, js_std, h_std, o_std, tdl_std))
# tot_samples = np.concatenate((samples, samples, samples, samples, samples, samples))
# print(tot_samples.T[0])
# print(tot_mean[0])
# print(tot_std[0])
# # print(tot_mean - tot_std)
#
# plt.rcParams.update({'font.size': 13.7})
# plt.subplots_adjust(left=0.15, bottom=0.11, right=0.9, top=0.87, wspace=0.22, hspace=0.35)
# plt.plot(tot_samples.T, tot_mean.T, marker='o')
# for i in range(6):
#     plt.fill_between(tot_samples[i], tot_mean[i] - tot_std[i], tot_mean[i] + tot_std[i], alpha=0.2)
# plt.legend(['Kolmorogov-Smirnov statistic', "Earth mover's distance", 'Jensen-Shannon distance', 'Hellinger distance',
#             'Overlapping coefficient', 'Time Difference Likelihood'])
# plt.xlabel('Number of samples')
# plt.ylabel('Time in seconds')
# plt.title('Computation time for various sample sizes')
# plt.show()
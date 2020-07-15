from __future__ import absolute_import
from __future__ import division

import random
import time
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt

from environments.climb_game import ClimbGame
from environments.extended_climb_game import ExtendedClimbGame
from environments.RO3 import R03Game

# Import algorithms
from algorithms.lmrl2 import LMRL2
from helper_functions.plotting import *
from algorithms.hysteretic_learning import Hysteretic

type = 'ps'  # det, ps, fs and with ExtendedClimbGame: normal
game = ExtendedClimbGame(type=type)
# game = ExtendedClimbGame(type=type)
# game = R03Game()
NUM_AGENTS = game.get_num_agents()
NUM_STATES = game.get_num_states()
NUM_ACTIONS = game.get_max_num_actions()
NUM_STATES_PLOT = 1
if isinstance(game, ClimbGame) or isinstance(game, ExtendedClimbGame):
    omega = 1
    if type == 'det':
        leniency = 10**7
        min_r = -30
        max_r = 11
    if type == 'ps':
        leniency = 10**3
        min_r = -30
        max_r = 14
    elif type == 'fs':
        leniency = 10
        min_r = -65
        max_r = 14
elif isinstance(game, R03Game):
    leniency = 1
    omega = 0.3
    min_r = 0
    max_r = 10


def main():
    # Parameters
    print_runs = False
    write_to_csv = True
    n_runs = 1
    iter_avg = 50
    num_episodes = 15000

    beta = 0.01  # alpha = 0.1, needed as parameter if some hysteretic version is used
    metric = 'ovl'  # options are dif_hist, ovl, ks, hellinger, jsd, emd, tdl
    e_decays = [0.9998]  # if 0.9998 it gets 0.018 in 10.000 runs and 0.135 in 5.000 runs
    t_decays = [0.999]
    # less than 0.9996 and more than 0.9999 does not make sense
    e_decays = [0.9996, 0.9997, 0.99975, 0.9998, 0.99985, 0.9999]
    t_decays = [0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.9925, 0.995, 0.996, 0.997, 0.9975, 0.998, 0.9985, 0.999]
    algo_name = 'll'  # ll, lsl, lhl, lhsl, hl, hsl, lsdl (only with similarity algorithms the metric is used)
    debug_run = 0

    correct_policy_results = np.zeros((len(t_decays), len(e_decays) + 1))
    sample_efficiency_mean_results = np.zeros((len(t_decays), len(e_decays) + 1))
    sample_efficiency_std_results = np.zeros((len(t_decays), len(e_decays) + 1))
    sample_efficiency_list_results = []

    for t_index, t_decay in enumerate(t_decays):
        correct_policy_results[t_index, 0] = t_decay
        sample_efficiency_mean_results[t_index, 0] = t_decay
        sample_efficiency_std_results[t_index, 0] = t_decay
        for e_index, e_decay in enumerate(e_decays):
            random.seed(1)
            correct_policies = 0
            rewards = []
            sim_metric = []
            delta_rec = []
            sample_efficiencies = []
            q_values = [[[[] for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES_PLOT)] for _ in range(NUM_AGENTS)]
            action_list = [[[[] for _ in range(NUM_ACTIONS)] for _ in range(NUM_STATES_PLOT)] for _ in range(NUM_AGENTS)]
            joint_actions = [[[] for _ in range(NUM_ACTIONS ** NUM_AGENTS)] for _ in range(NUM_STATES_PLOT)]
            sim_met_per_j_a = [[[] for _ in range(NUM_ACTIONS ** NUM_AGENTS)] for _ in range(NUM_STATES_PLOT)]
            j_a_dict = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 0): 3, (1, 1): 4, (1, 2): 5, (2, 0): 6, (2, 1): 7, (2, 2): 8}

            algo = LMRL2(NUM_STATES, beta, leniency, e_decay, t_decay, min_r, max_r, game,
                         algo_name, metric=metric, init=min_r)
            # Training
            for run in tqdm(range(n_runs)):
                algo.reset_values()
                sample_counter = 0
                for i in range(num_episodes):
                    next_state = game.new_game()

                    while not game.is_terminal():
                        cur_state = next_state
                        poss_actions = game.possible_actions(cur_state)
                        a_1 = algo.select_action(poss_actions, cur_state, 0)
                        a_2 = algo.select_action(poss_actions, cur_state, 1)
                        actions = (a_1, a_2)
                        next_state, reward = game.next_step(cur_state, actions)
                        algo.next_step(actions, cur_state, next_state, reward)

                        # Collect values for plotting
                        collect_values_for_plotting(cur_state, next_state, reward, rewards, delta_rec, q_values, algo,
                                                    action_list, j_a_dict, actions, joint_actions, sim_metric,
                                                    sim_met_per_j_a, algo_name, NUM_AGENTS)

                    if np.argmax(algo.q_values[0][0]) == 0 and np.argmax(algo.q_values[1][0]) == 0:
                        sample_counter = sample_counter + 1
                    else:
                        sample_counter = 0

                # Calculate correct policies
                if np.argmax(algo.q_values[0][0]) == 0 and np.argmax(algo.q_values[1][0]) == 0:
                    if isinstance(game, R03Game):
                        if np.argmax(algo.q_values[0][1]) == np.argmax(algo.q_values[1][1]):
                            correct_policies += 1
                    elif isinstance(game, ClimbGame) or isinstance(game, ExtendedClimbGame):
                        correct_policies += 1
                        sample_efficiencies.append(num_episodes - sample_counter)
                elif print_runs:  # print if policy is not correct
                    print(run)

            print(f'e_decay: {e_decay}, t_decay: {t_decay}')
            print(f'sample efficiencies: {sample_efficiencies}')
            if len(sample_efficiencies) > 0:
                se_mean = np.mean(sample_efficiencies)
                se_std = np.std(sample_efficiencies)
            else:
                se_mean = 15000
                se_std = 0
            print(f'mean sample efficiency: {se_mean}, std: {se_std}')
            print(f'correct policies: {correct_policies / n_runs}\n')
            correct_policy_results[t_index, e_index + 1] = correct_policies / n_runs * 100
            sample_efficiency_mean_results[t_index, e_index + 1] = np.round(se_mean, decimals=0)
            sample_efficiency_std_results[t_index, e_index + 1] = se_std
            sample_efficiency_list_results.append([t_decay, e_decay] + sample_efficiencies)

        # Writing results to csv file
        if write_to_csv:
            fields = ['Temp decay'] + e_decays
            filename = f'cp_{algo_name}_{metric}_{num_episodes}.csv'
            with open(filename, 'w') as csvfile_cp:
                csvwriter_cp = csv.writer(csvfile_cp)
                csvwriter_cp.writerow(fields)
                csvwriter_cp.writerows(correct_policy_results)
            filename = f'se_mean_{algo_name}_{metric}_{num_episodes}.csv'
            with open(filename, 'w') as csvfile_se_mean:
                csvwriter_se_mean = csv.writer(csvfile_se_mean)
                csvwriter_se_mean.writerow(fields)
                csvwriter_se_mean.writerows(sample_efficiency_mean_results)
            filename = f'se_std_{algo_name}_{metric}_{num_episodes}.csv'
            with open(filename, 'w') as csvfile_se_std:
                csvwriter_se_std = csv.writer(csvfile_se_std)
                csvwriter_se_std.writerow(fields)
                csvwriter_se_std.writerows(sample_efficiency_std_results)
            filename = f'se_list_{algo_name}_{metric}_{num_episodes}.csv'
            with open(filename, 'w') as csvfile_se_list:
                csvwriter_se_list = csv.writer(csvfile_se_list)
                csvwriter_se_list.writerows(sample_efficiency_list_results)

    # plotting_all(algo, algo_name, j_a_dict, sim_metric, sim_met_per_j_a, q_values, joint_actions, action_list,
    #              delta_rec, rewards, game, NUM_AGENTS, NUM_ACTIONS, iter_avg, n_runs, num_episodes, font_size=9)

    # Plotting for report (use the size that is directly given, do not rescale plot)
    plt.rcParams.update({'font.size': 14})
    plt.figure()
    qvalue_plot(q_values, 0, game, NUM_AGENTS, NUM_ACTIONS, iter_avg, n_runs, num_episodes, run=debug_run)
    plt.subplots_adjust(left=0.1, bottom=0.11, right=0.9, top=0.94, wspace=0.22, hspace=0.35)
    plt.show()
    plt.figure()
    action_plot(action_list, 0, NUM_AGENTS, NUM_ACTIONS, iter_avg, n_runs, num_episodes, run=debug_run)
    plt.subplots_adjust(left=0.11, bottom=0.11, right=0.90, top=0.94, wspace=0.22, hspace=0.35)
    plt.show()
    # phase_plot(action_list, 0, iter_avg, n_runs, num_episodes, NUM_AGENTS)
    # plt.show()


if __name__ == "__main__":
    main()

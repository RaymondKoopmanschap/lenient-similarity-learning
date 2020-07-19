from __future__ import absolute_import
from __future__ import division

import random
from tqdm import tqdm
from argparse import ArgumentParser

# Import algorithms
from algorithms.lmrl2 import LMRL2
from helper_functions.plotting import *
from helper_functions.helper_functions import *

NUM_AGENTS = 2  # 2 agents is the only option for now
NUM_STATES_PLOT = 1  # 1 is the only option for now


def main(args):
    game_type = args.game_type
    if args.game == "CB":
        game = ClimbGame(game_type=game_type)
    elif args.game == "ECB":
        game = ExtendedClimbGame(game_type=game_type)
    elif args.game == "RO3":
        game = R03Game()
    else:
        print("Choose a valid game: CB, ECB or RO3")
        game = None

    num_states = game.get_num_states()
    num_actions = game.get_max_num_actions()
    min_r, max_r, leniency, omega = get_lenient_parameters(game, game_type)
    # Parameters for the run
    print_runs = args.print_runs
    write_to_csv_bool = args.write_to_csv
    n_runs = args.n_runs
    iter_avg = args.iter_avg
    num_episodes = args.num_episodes

    # Parameters for the algorithm
    beta = args.beta  # alpha = 0.1, needed as parameter if some hysteretic version is used
    metric = args.sim_metric  # options are dif_hist, ovl, ks, hellinger, jsd, emd, tdl
    e_decays = [args.e_decays]  # if 0.9998 it gets 0.018 in 10.000 runs and 0.135 in 5.000 runs
    t_decays = [args.t_decays]
    if args.grid_search == 'small':
        e_decays = [0.999, 0.9991, 0.9992, 0.9993, 0.9994, 0.9995]
        t_decays = [0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.9925, 0.995]
    elif args.grid_search == 'large':
        e_decays = [0.9991, 0.9992, 0.9993, 0.9994, 0.9995, 0.9996, 0.9997, 0.99975, 0.9998, 0.99985, 0.9999]
        t_decays = [0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.9925, 0.995, 0.996, 0.997, 0.9975, 0.998, 0.9985,
                    0.999]
    algo_name = args.algo_name  # ll, lsl, lhl, lhsl, hl, hsl, lsdl (only with similarity algorithms the metric is used)
    debug_run = args.debug_run

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
            q_values = [[[[] for _ in range(num_actions)] for _ in range(NUM_STATES_PLOT)] for _ in range(NUM_AGENTS)]
            action_list = [[[[] for _ in range(num_actions)] for _ in range(NUM_STATES_PLOT)] for _ in range(NUM_AGENTS)]
            joint_actions = [[[] for _ in range(num_actions ** NUM_AGENTS)] for _ in range(NUM_STATES_PLOT)]
            sim_met_per_j_a = [[[] for _ in range(num_actions ** NUM_AGENTS)] for _ in range(NUM_STATES_PLOT)]
            j_a_dict = {(0, 0): 0, (0, 1): 1, (0, 2): 2, (1, 0): 3, (1, 1): 4, (1, 2): 5, (2, 0): 6, (2, 1): 7, (2, 2): 8}

            algo = LMRL2(num_states, beta, leniency, e_decay, t_decay, min_r, max_r, game,
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
            print(f'correct policies: {correct_policies / n_runs * 100}%\n')
            correct_policy_results[t_index, e_index + 1] = correct_policies / n_runs * 100
            sample_efficiency_mean_results[t_index, e_index + 1] = np.round(se_mean, decimals=0)
            sample_efficiency_std_results[t_index, e_index + 1] = se_std
            sample_efficiency_list_results.append([t_decay, e_decay] + sample_efficiencies)

        # Writing results to csv file
        if write_to_csv_bool:
            write_to_csv(e_decays, algo_name, metric, num_episodes, correct_policy_results, game_type,
                         sample_efficiency_mean_results, sample_efficiency_std_results, sample_efficiency_list_results)

    # plotting_all(algo, algo_name, j_a_dict, sim_metric, sim_met_per_j_a, q_values, joint_actions, action_list,
    #              delta_rec, rewards, game, NUM_AGENTS, num_actions, iter_avg, n_runs, num_episodes, font_size=9)

    # Plotting for report (use the size that is directly given, do not rescale plot)
    plt.rcParams.update({'font.size': 14})
    plt.figure()
    qvalue_plot(q_values, 0, game, NUM_AGENTS, num_actions, iter_avg, n_runs, num_episodes, run=debug_run)
    plt.ylim(bottom=0, top=15)
    plt.subplots_adjust(left=0.1, bottom=0.11, right=0.9, top=0.94, wspace=0.22, hspace=0.35)
    plt.show()
    plt.figure()
    action_plot(action_list, 0, NUM_AGENTS, num_actions, iter_avg, n_runs, num_episodes, run=debug_run)
    plt.subplots_adjust(left=0.11, bottom=0.11, right=0.90, top=0.94, wspace=0.22, hspace=0.35)
    plt.show()
    # phase_plot(action_list, 0, iter_avg, n_runs, num_episodes, NUM_AGENTS)
    # plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-g", "--grid_search", type=str, help="parameter to use the same grid search as is used in the "
                                                              "experiments (this replaces e_decays and t_decays args). "
                                                              "Choose between: small and large")
    parser.add_argument("-e", "--e_decays", type=float,
                        help="give the epsilon decay parameter to use")
    parser.add_argument("-t", "--t_decays", type=float,
                        help="give the temperature decay parameter to use")
    parser.add_argument("--n_runs", type=int, help="number of runs to execute", default=100)
    parser.add_argument("--num_episodes", type=int, help="number of episodes per run", default=15000)
    parser.add_argument("--print_runs", type=bool, help="Print runs or not", default=False)
    parser.add_argument("--write_to_csv", type=bool, help="Write results to csv or not", default=True)
    parser.add_argument("--iter_avg", type=int, help="It takes the average of iter_avg number of iterations to help "
                                                     "smooth out the plots", default=50)
    parser.add_argument("--game_type", type=str, help="Choose game type: det, ps or fs, not needed for RO3",
                        default='ps')
    parser.add_argument("--game", type=str, help="Choose game: CB, ECB or RO3", default='ECB')
    parser.add_argument("--beta", type=float, help="Choose beta needed for hysteretic learning versions", default=0.01)
    parser.add_argument("--sim_metric", type=str, help="Choose a similarity metric: ovl, emd, ks, tdl, hellinger, jsd",
                        default='ovl')
    parser.add_argument("--algo_name", type=str,
                        help="Choose which algorithm to use: ll, lsl, lhl, lhsl, hl, hsl, lsdl", default='ll')
    parser.add_argument("--debug_run", type=int, help="Choose to show a specific run in the plot. The number of runs "
                                                      "has to go until that number", default=None)
    args = parser.parse_args()
    main(args)

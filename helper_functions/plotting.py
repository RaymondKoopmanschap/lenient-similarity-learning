import math
import numpy as np
import matplotlib.pyplot as plt
import ternary
from environments.RO3 import R03Game
from environments.climb_game import ClimbGame
from scipy.stats import norm
import matplotlib


def plotting_all(algo_name, j_a_dict, sim_metric, sim_met_per_j_a, q_values, joint_actions, action_list,
                 delta_rec, rewards, game, num_agents, num_actions, iter_avg, n_runs, num_episodes, interval_plotting,
                 beta, font_size):
    plt.rcParams.update({'font.size': font_size})  # change to 14 for report
    state = 0
    if 's' in algo_name:
        cols = 4
        plt.subplot(2, cols, 7)
        sim_metric_plot(sim_metric, iter_avg, n_runs, num_episodes)
        plt.subplot(2, cols, 8)
        plot_sim_metric_j_a(sim_met_per_j_a, j_a_dict, state, iter_avg, n_runs, num_episodes)
    else:
        cols = 3
    plt.subplot(2, cols, 1)
    qvalue_plot(q_values, state, game, num_agents, num_actions, iter_avg, n_runs, num_episodes, interval_plotting,
                algo_name, beta)
    plt.subplot(2, cols, 2)
    joint_action_plot(joint_actions, j_a_dict, state, iter_avg, n_runs, num_episodes)
    plt.subplot(2, cols, 3)
    action_plot(action_list, state, num_agents, num_actions, iter_avg, n_runs, num_episodes, interval_plotting)
    plt.subplot(2, cols, 4)
    plot_delta(delta_rec)
    plt.subplot(2, cols, 5)
    plot_perc_miscoordination(joint_actions, state, iter_avg, n_runs, num_episodes)
    plt.subplot(2, cols, 6)
    avg_reward_plot(rewards, game, iter_avg, n_runs, num_episodes)
    # phase_plot(action_list, state, iter_avg, n_runs, num_episodes, num_agents)
    plt.show()


def collect_values_for_plotting(cur_state, next_state, reward, rewards, delta_rec, q_values, algo, action_list,
                                j_a_dict, actions, joint_actions, sim_metric, sim_met_per_j_a, algo_name, num_agents):
    if next_state == 'terminal':
        rewards.append(reward)

    if cur_state == 0:
        delta_rec.append(algo.delta_rec)
        for agent in range(num_agents):
            for action, q_value in enumerate(algo.q_values[agent][cur_state]):
                q_values[agent][cur_state][action].append(q_value)
                if action == actions[agent]:
                    action_list[agent][cur_state][action].append(1)
                else:
                    action_list[agent][cur_state][action].append(0)
        for count, joint_action in enumerate(j_a_dict):
            if joint_action == actions:
                joint_actions[cur_state][count].append(1)
            else:
                joint_actions[cur_state][count].append(0)

        if 's' in algo_name:
            if algo.delta_rec < 0:
                sim_metric.append(algo.sim_metric)
            else:
                sim_metric.append(math.nan)
            for count, joint_action in enumerate(j_a_dict):
                if joint_action == actions:
                    if algo.delta_rec < 0:
                        sim_met_per_j_a[cur_state][count].append(algo.sim_metric)
                    else:
                        sim_met_per_j_a[cur_state][count].append(math.nan)
                else:
                    sim_met_per_j_a[cur_state][count].append(math.nan)


def plot_standard_deviation(means, std, x_axis, state, num_agents, num_actions):
    for agent in range(num_agents):
        for action in range(num_actions):
            plt.fill_between(
                x_axis,
                means[agent, state, action] + std[agent, state, action],
                means[agent, state, action] - std[agent, state, action],
                alpha=0.2)


def shape_dim_for_plot(iter_avg, n_runs, data, run):
    """Go from shape: (:, data) → (-1, iter_avg) → (-1) (average over data) → (n_runs, 250)
    Example: (5000,) → (250, 20) → (250, ) → (1, 250)"""
    data = np.asarray(data)
    shape_other_dims = data.shape[:-1]
    data = data.reshape(shape_other_dims + (-1, iter_avg))
    avg_data = np.nanmean(data, axis=-1)
    if run is not None:
        avg_data = avg_data.reshape(shape_other_dims + (n_runs, -1))
        avg_data = avg_data[..., run, :]
        last_dim = avg_data.shape[-1]
        avg_data = avg_data.reshape(shape_other_dims + (1, last_dim))
    else:
        avg_data = avg_data.reshape(shape_other_dims + (n_runs, -1))
    return avg_data


def plot_2agent_3actions(x_axis, means, state, num_episodes, iter_avg, interval):
    markers_on1 = np.linspace(start=0, stop=(num_episodes/iter_avg)-interval/iter_avg,
                             num=int(num_episodes/interval)).astype(int).tolist()
    step_size = markers_on1[1] - markers_on1[0]
    shift_1 = step_size / 6 * 1
    shift_2 = step_size / 6 * 2
    shift_3 = step_size / 6 * 3
    shift_4 = step_size / 6 * 4
    shift_5 = step_size / 6 * 5
    markers_on2 = np.linspace(start=0 + shift_1, stop=(num_episodes / iter_avg) + shift_1 - interval / iter_avg,
                              num=int(num_episodes / interval)).astype(int).tolist()
    markers_on3 = np.linspace(start=0 + shift_2, stop=(num_episodes / iter_avg) + shift_2 - interval / iter_avg,
                              num=int(num_episodes / interval)).astype(int).tolist()
    markers_on4 = np.linspace(start=0 + shift_3, stop=(num_episodes / iter_avg) + shift_3 - interval / iter_avg,
                              num=int(num_episodes / interval)).astype(int).tolist()
    markers_on5 = np.linspace(start=0 + shift_4, stop=(num_episodes / iter_avg) + shift_4 - interval / iter_avg,
                              num=int(num_episodes / interval)).astype(int).tolist()
    markers_on6 = np.linspace(start=0 + shift_5, stop=(num_episodes / iter_avg) + shift_5 - interval / iter_avg,
                              num=int(num_episodes / interval)).astype(int).tolist()
    markersize = 5
    linewidth = 1
    color_1 = 'darkturquoise'
    color_2 = '#e67e00'
    # marker='^', markersize=markersize, markevery=markers_on1, markerfacecolor=color_1, markeredgecolor=color_1
    plt.plot(x_axis, means[0, state, 0], label="Agent 1 action A", color='cyan', linewidth=linewidth, zorder=1,
             marker='o', markersize=markersize, markevery=markers_on1, markerfacecolor=color_1, markeredgecolor=color_1)
    plt.plot(x_axis, means[0, state, 1], label="Agent 1 action B", color='cyan', linewidth=linewidth, zorder=1,
             marker='^', markersize=markersize, markevery=markers_on2, markerfacecolor=color_1, markeredgecolor=color_1)
    plt.plot(x_axis, means[0, state, 2], label="Agent 1 action C", color='cyan', linewidth=linewidth, zorder=1,
             marker='s', markersize=markersize, markevery=markers_on3, markerfacecolor=color_1, markeredgecolor=color_1)
    plt.plot(x_axis, means[1, state, 0], label="Agent 2 action A", color='orange', linewidth=linewidth, zorder=1,
             marker='o', markersize=markersize, markevery=markers_on4, markerfacecolor=color_2, markeredgecolor=color_2)
    plt.plot(x_axis, means[1, state, 1], label="Agent 2 action B", color='orange', linewidth=linewidth, zorder=1,
             marker='^', markersize=markersize, markevery=markers_on5, markerfacecolor=color_2, markeredgecolor=color_2)
    plt.plot(x_axis, means[1, state, 2], label="Agent 2 action C", color='orange', linewidth=linewidth, zorder=1,
             marker='s', markersize=markersize, markevery=markers_on6, markerfacecolor=color_2, markeredgecolor=color_2)
    plt.scatter(x_axis[markers_on1], means[0, state, 0][markers_on1], color=color_1, marker='o', zorder=2)
    plt.scatter(x_axis[markers_on2], means[0, state, 1][markers_on2], color=color_1, marker='^', zorder=2)
    plt.scatter(x_axis[markers_on3], means[0, state, 2][markers_on3], color=color_1, marker='s', zorder=2)
    plt.scatter(x_axis[markers_on4], means[1, state, 0][markers_on4], color=color_2, marker='o', zorder=2)
    plt.scatter(x_axis[markers_on5], means[1, state, 1][markers_on5], color=color_2, marker='^', zorder=2)
    plt.scatter(x_axis[markers_on6], means[1, state, 2][markers_on6], color=color_2, marker='s', zorder=2)
    plt.legend(prop={'size': 10})


def qvalue_plot(q_values, state, game, num_agents, num_actions, iter_avg, n_runs, num_episodes, interval_plotting,
                algo_name, beta, plot_std=False, run=None):
    avg_q_values = shape_dim_for_plot(iter_avg, n_runs, q_values, run)
    x_axis = np.arange(0, num_episodes/iter_avg) * iter_avg
    means = avg_q_values.mean(axis=3)
    plot_2agent_3actions(x_axis, means, state, num_episodes, iter_avg, interval_plotting)
    if plot_std:
        std = avg_q_values.std(axis=3)
        plot_standard_deviation(means, std, x_axis, state, num_agents, num_actions)
    linewidth = 0.5
    margin = 0.05
    offset = 1000
    if isinstance(game, ClimbGame):
        if algo_name == 'hl' and beta == 0.1:
            plt.axhline(y=-6.33, linewidth=linewidth, color='grey')
            plt.axhline(y=-5.67, linewidth=linewidth, color='grey')
            plt.axhline(y=1.67, linewidth=linewidth, color='grey')
            plt.axhline(y=-7.67, linewidth=linewidth, color='grey')
            plt.axhline(y=3.67, linewidth=linewidth, color='grey')
            plt.axvline(x=100, linewidth=linewidth, color='grey')
            plt.text(x=offset, y=-6.33 + margin, s='-6.33')
            plt.text(x=offset, y=-5.67 + margin, s='-5.67')
            plt.text(x=offset, y=1.67 + margin, s='1.67')
            plt.text(x=offset, y=-7.67 + margin, s='-7.67')
            plt.text(x=offset, y=3.67 + margin, s='3.67')
        else:
            plt.axhline(y=11, linewidth=linewidth, color='grey')
            # plt.axhline(y=7, linewidth=linewidth, color='grey')
            plt.text(x=-200, y=11 + margin, s='11')
            # plt.text(x=-200, y=7 + margin, s='7')

    elif isinstance(game, R03Game):
        plt.axhline(y=5, linewidth=linewidth, color='grey')
    plt.title(f'Q-values of each agent per {iter_avg} time steps')
    plt.xlabel("number of iterations")
    plt.ylabel("return")


def action_plot(action_list, state, num_agents, num_actions, iter_avg, n_runs, num_episodes, interval_plotting,
                plot_std=False, run=None):
    perc_actions = shape_dim_for_plot(iter_avg, n_runs, action_list, run)
    means = perc_actions.mean(axis=3) * 100  # convert to percentage
    x_axis = np.arange(0, num_episodes/iter_avg) * iter_avg
    plot_2agent_3actions(x_axis, means, state, num_episodes, iter_avg, interval_plotting)
    if plot_std:
        std = perc_actions.std(axis=3)
        plot_standard_deviation(means, std, x_axis, state, num_agents, num_actions)
    plt.title(f'Percentage of actions per {iter_avg} time steps')
    plt.xlabel("number of iterations")
    plt.ylabel("percentage")


def sim_metric_plot(sim_metric, iter_avg, n_runs, num_episodes, plot_std=False, run=None):
    avg_sim_metric = shape_dim_for_plot(iter_avg, n_runs, sim_metric, run)
    mean_runs = np.nanmean(avg_sim_metric, axis=0)
    x_axis = np.arange(0, num_episodes / iter_avg) * iter_avg
    plt.scatter(x_axis, mean_runs, s=1)
    plt.plot(x_axis, mean_runs, label="similarity metric value")
    if plot_std:
        std_runs = avg_sim_metric.std(axis=0)
        plt.fill_between(x_axis, mean_runs + std_runs, mean_runs - std_runs, alpha=0.2)
    plt.title(f'Average similarity metric value per {iter_avg} time steps')
    plt.xlabel('number of iterations')
    plt.ylabel('similarity metric value')
    plt.legend()


def plot_sim_metric_j_a(sim_met_per_j_a, j_a_dict, state, iter_avg, n_runs, num_episodes, run=None):
    avg_sim_metric = shape_dim_for_plot(iter_avg, n_runs, sim_met_per_j_a, run)
    means = np.nanmean(avg_sim_metric, axis=2)
    x_axis = np.arange(0, num_episodes / iter_avg) * iter_avg
    for count, joint_action in enumerate(j_a_dict):
        plt.scatter(x_axis, means[state][count], s=1)
        plt.plot(x_axis, means[state][count], label="action " + str(joint_action))
    plt.title(f'Joint-action similarity values per {iter_avg} time steps')
    plt.xlabel('number of iterations')
    plt.ylabel('similarity metric value')
    plt.legend()


def avg_reward_plot(rewards, game, iter_avg, n_runs, num_episodes, plot_std=True, run=None):
    avg_rewards = shape_dim_for_plot(iter_avg, n_runs, rewards, run)
    mean_runs = avg_rewards.mean(axis=0)
    x_axis = np.arange(0, num_episodes/iter_avg) * iter_avg
    plt.plot(x_axis, mean_runs, label="average reward per " + str(iter_avg) + " runs")
    if plot_std:
        std_runs = avg_rewards.std(axis=0)
        plt.fill_between(x_axis, mean_runs + std_runs, mean_runs - std_runs, alpha=0.2)
    if isinstance(game, ClimbGame):
        plt.axhline(y=11, linewidth=0.5, color='grey')
    elif isinstance(game, R03Game):
        plt.axhline(y=5, linewidth=0.5, color='grey')

    plt.title(f'Average reward per {iter_avg} time steps')
    plt.xlabel('number of iterations')
    plt.ylabel('reward per time step')
    plt.legend()


def joint_action_plot(joint_actions, j_a_dict, state, iter_avg, n_runs, num_episodes, run=None):
    x_axis = np.arange(0, num_episodes/iter_avg) * iter_avg
    perc_joint_actions = shape_dim_for_plot(iter_avg, n_runs, joint_actions, run)
    means = perc_joint_actions.mean(axis=2)
    for count, joint_action in enumerate(j_a_dict):
        plt.plot(x_axis, means[state][count], label="action " + str(joint_action))
    plt.title(f'Percentage of joint-actions per {iter_avg} time steps')
    plt.xlabel("number of iterations")
    plt.ylabel("percentage")
    plt.legend()


def phase_plot(action_list, state, iter_avg, n_runs, num_episodes, num_agents, run=None):
    perc_actions = shape_dim_for_plot(iter_avg, n_runs, action_list, run)
    means = perc_actions.mean(axis=3)
    phase_index = [[] for _ in range(num_agents)]
    for agent in range(num_agents):
        for sample in range(int(num_episodes/iter_avg)):
            perc_c = means[agent, state, 2][sample]
            perc_a = means[agent, state, 0][sample]
            phase_index[agent].append((perc_c, perc_a))
    figure, tax = ternary.figure(scale=1)
    tax.plot(phase_index[0], label="Agent 1")
    tax.plot(phase_index[1], label="Agent 2")
    fontsize = 14
    tax.top_corner_label("A", fontsize=fontsize)
    tax.right_corner_label("C", fontsize=fontsize)
    tax.left_corner_label("B", fontsize=fontsize)
    tax.gridlines(multiple=0.2, color="black")
    tax.set_title("Trajectory distributions\n", fontsize=14)
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    tax.legend()


def plot_perc_miscoordination(joint_actions, state, iter_avg, n_runs, num_episodes, run=None):
    perc_joint_actions = shape_dim_for_plot(iter_avg, n_runs, joint_actions, run)
    means = perc_joint_actions.mean(axis=2)
    x_axis = np.arange(0, num_episodes/iter_avg) * iter_avg
    perc_coor_1 = means[state][0] + means[state][4] + means[state][8]
    perc_coor_2 = means[state][0] + means[state][4] + means[state][5]
    perc_misc_1 = 1 - perc_coor_1
    perc_misc_2 = 1 - perc_coor_2
    plt.plot(x_axis, perc_misc_1, label='agent 1')
    plt.plot(x_axis, perc_misc_2, label='agent 2')
    plt.title(f"Percentage miscoordination per {iter_avg} time steps")
    plt.xlabel("number of iterations")
    plt.ylabel("percentage")
    plt.legend()


def plot_delta(delta):
    plt.plot(delta, label='delta')
    plt.title('delta value for each time step')
    plt.xlabel('number of iterations')
    plt.ylabel('delta')
    plt.legend()


def kolmogorov_smirnov_plot():
    np.random.seed(3)
    size = 30
    data1 = np.random.normal(loc=0, scale=10.0, size=size+1)
    data2 = np.random.normal(loc=0, scale=10.0, size=size+1)
    data1.sort(), data2.sort()

    tot_dist = np.concatenate([data1, data2])
    cdf1 = np.searchsorted(data1, tot_dist, side='right') / size
    cdf2 = np.searchsorted(data2, tot_dist, side='right') / size
    loc = np.argmax(abs(cdf1 - cdf2))
    val = tot_dist[loc] + 0.2
    loc_cdf1 = np.searchsorted(data1, val, side='right') / size
    loc_cdf2 = np.searchsorted(data2, val, side='right') / size
    min_loc = min(loc_cdf1, loc_cdf2)
    max_loc = max(loc_cdf1, loc_cdf2)

    # Cumulative distributions, stepwise:
    axes1 = plt.step(data1, np.arange(data1.size)/size, label='empirical cdf $F_1(x)$')
    axes2 = plt.step(data2, np.arange(data2.size)/size, label='empirical cdf $F_2(x)$')
    matplotlib.rcParams.update({'font.size': 22})
    plt.annotate(s='', xy=(val, min_loc), xytext=(val, max_loc), arrowprops=dict(arrowstyle='<->'))
    plt.text(x=val, y=(min_loc+max_loc)/2, s='$D$', fontsize=20)
    plt.text(x=-25.9, y=0.83, s='$D = sup_x|F_1(x) - F_2(x)|$')

    plt.title('Two-sampled Kolmogorov-Smirnov test statistic $D$')
    plt.ylabel('Probability', fontsize=22)
    plt.xlabel('Value', fontsize=22)
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()


def multi_modal_plot_for_tdl():
    # Multi-modal stuff
    mu_1 = 0
    mu_2 = 10
    mu_3 = 5
    variance = 0.5
    sigma = np.sqrt(variance)
    x_1 = np.linspace(mu_1 - 5*sigma, mu_1 + 7*sigma, 100)
    x_2 = np.linspace(mu_2 - 7*sigma, mu_2 + 5*sigma, 100)
    x_3 = np.linspace(mu_3 - 7*sigma, mu_3 + 7*sigma, 100)
    plt.rcParams.update({'font.size': 14})
    plt.plot(x_1, norm.pdf(x_1, mu_1, sigma)/2, color='C1', label='target distribution')
    plt.plot(x_2, norm.pdf(x_2, mu_2, sigma)/2, color='C1')
    plt.plot(x_3, norm.pdf(x_3, mu_3, sigma), color='C0', label='base distribution')
    plt.title('Multi-modal and normal distribution')
    plt.xlabel('value')
    plt.ylabel('probability')
    plt.legend()
    plt.show()

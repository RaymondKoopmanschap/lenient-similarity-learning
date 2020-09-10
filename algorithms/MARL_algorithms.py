import math
import matplotlib.pyplot as plt
from algorithms.similarity_metrics import *


class MARLAlgorithms(object):

    def __init__(self, states, beta, leniency, e_decay, t_decay,
                 min_r, max_r, game, algo_name, n_samples, metric='dif_hist', init=0):
        # parameters
        self.num_pids = 2  # player ids
        self.num_a = 3
        self.num_states = states
        self.alpha = 0.1
        self.beta = beta
        self.gamma = 1
        self.delta_rec = None
        self.epsilon = None
        self.epsilon_decay = e_decay
        self.min_r = int(min_r)
        self.max_r = int(max_r)
        self.q_values = None
        self.game = game
        self.init = init
        self.algo = algo_name

        # Leniency parameters
        self.t_decay = t_decay
        self.MaxTemp = 50
        self.MinTemp = 2
        self.omega = 1
        self.theta = leniency  # leniency moderator factor
        self.temp_diffusion = 0.1
        self.state_action_map = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 0),
                                 5: (1, 1), 6: (1, 2), 7: (2, 0), 8: (2, 1), 9: (2, 2)}
        # For now use the same temperature for every state (as if there was one state)
        self.t_values = None
        self.return_dist = None
        self.dist_id = None
        self.n_samples = n_samples
        self.sim_metric = None
        self.metric_name = metric
        if 's' in self.algo:
            self.dist = True
        else:
            self.dist = False
        self.bins = np.linspace(self.min_r, self.max_r + 0.000001, int(self.max_r - self.min_r) + 1)  # + 1

        # For collecting values
        self.j_a_0_0 = [[[], []], [[], []]]  # first index is agent number, second: sim value or iteration
        self.j_a_0_1 = [[[], []], [[], []]]
        self.j_a_0_2 = [[[], []], [[], []]]
        self.j_a_1_0 = [[[], []], [[], []]]
        self.j_a_1_1 = [[[], []], [[], []]]
        self.j_a_1_2 = [[[], []], [[], []]]
        self.j_a_2_0 = [[[], []], [[], []]]
        self.j_a_2_1 = [[[], []], [[], []]]
        self.j_a_2_2 = [[[], []], [[], []]]

    def reset_values(self):
        """
        Reset the values after run completed
        """
        self.q_values = [[[self.init for _ in range(self.num_a)]
                          for _ in range(self.num_states)] for _ in range(self.num_pids)]
        self.t_values = [[[self.MaxTemp for _ in range(self.num_a)]
                          for _ in range(self.num_states)] for _ in range(self.num_pids)]

        self.epsilon = 1
        self.delta_rec = 0
        self.return_dist = [[[[] for _ in range(self.num_a)] for _ in range(self.num_states)]
                            for _ in range(self.num_pids)]
        self.dist_id = [[[0 for _ in range(self.num_a)] for _ in range(self.num_states)]
                        for _ in range(self.num_pids)]
        self.sim_metric = 0

    def select_action(self, poss_a, state, pid_id):
        """
        Selects an action for a specific pid.
        :param poss_a: possible actions
        :param state: the state of the agent
        :param pid_id: the agent id
        :return: action
        """
        # Epsilon-greedy
        rand = random.random()
        if rand < self.epsilon:
            action = random.choice(poss_a)
        else:
            action = self.q_values[pid_id][state].index(max(self.q_values[pid_id][state][0:len(poss_a)]))
        return action

    def next_step(self, a, s, next_s, r, i, debug_run, run, vis_iters, vis_pids):
        """
        Execute a single RL step
        :param a: action
        :param s: state
        :param next_s: next state
        :param r: reward
        :param i: iteration
        :param debug_run: specify if I want to show plots for a specific run
        :param run: run number
        :param vis_iters: the iterations I want to visualize for the current and target distribution
        :param vis_pids: for which agent I want to visualize these iterations
        :return: nothing
        """
        self.epsilon = self.epsilon * self.epsilon_decay
        for pid in range(self.num_pids):
            # Calculating delta
            if next_s == 'terminal' or max(self.q_values[pid][next_s]) == math.inf:
                delta = r - self.q_values[pid][s][a[pid]]
            else:
                delta = r + self.gamma * max(self.q_values[pid][next_s]) - self.q_values[pid][s][a[pid]]
            self.delta_rec = delta

            # Calculating the similarity metric
            if self.dist:
                self.calculate_similarity(a, s, next_s, r, pid, delta, i, vis_iters, vis_pids, run, debug_run)
            else:
                self.sim_metric = None

            if delta < 0 and s == 0 and run == debug_run:
                self.collecting_similarity_values_for_plot(pid, a, i)

            """ the code below can indicate seven different algorithms
                lenient learning:            (ll)   if alpha_sim = alpha      and prob = temp prob
                lenient similarity learning: (lsl)  if alpha_sim = alpha      and prob = l
                lenient hysteretic learning: (lhl)  if alpha_sim = 0          and prob = temp prob
                lenient hyst. sim. learning: (lhsl) if alpha_sim = alpha * l  and prob = temp prob
                hysteretic learning:         (hl)   if alpha_sim = 0          and prob = 1 
                hysteretic sim. learning:    (hsl)  if alpha_sim = alpha * l  and prob = 1 (change ret dist)
                lenient sim. Daan learning:  (lsdl) if alpha_sim = alpha      and prob = min(l, temp prob)"""

            alpha_sim = self.calculate_alpha_sim()
            prob = self.calculate_prob(a, s, pid)  # uses the similarity metric
            rand = random.random()

            # The Q-learning update
            if delta >= 0 or rand < prob:  # the lenient learning check (higher prob more likely to update)
                if delta >= 0:  # the hysteretic learning check
                    self.q_values[pid][s][a[pid]] = self.q_values[pid][s][a[pid]] + self.alpha * delta
                else:
                    self.q_values[pid][s][a[pid]] = self.q_values[pid][s][a[pid]] + max(self.beta, alpha_sim) * delta
            elif self.q_values[pid][s][a[pid]] == math.inf:
                self.q_values[pid][s][a[pid]] = r
            else:
                pass

            # Temperature update
            if next_s != 'terminal':
                self.t_values[pid][s][a[pid]] = self.t_decay * ((1 - self.temp_diffusion)*self.t_values[pid][s][a[pid]]
                                              + self.temp_diffusion*np.mean(self.t_values[pid][next_s][0:1]))
                # THIS [0:1] IS ONLY FOR THE EXTENDED CLIMB GAME AS AN EASY FIX !
            else:
                self.t_values[pid][s][a[pid]] = self.t_decay * self.t_values[pid][s][a[pid]]

            # Building up the return distributions
            if self.dist:
                if delta >= 0 or rand < prob:
                    if next_s != 'terminal':
                        ret = r + self.gamma * max(self.q_values[pid][next_s])
                    else:
                        ret = self.q_values[pid][s][a[pid]]  # r
                    if len(self.return_dist[pid][s][a[pid]]) < self.n_samples:
                        self.return_dist[pid][s][a[pid]].append(ret)
                    # List is complete and replace the oldest value
                    else:
                        self.return_dist[pid][s][a[pid]][self.dist_id[pid][s][a[pid]]] = ret
                        if self.dist_id[pid][s][a[pid]] < (self.n_samples - 1):
                            self.dist_id[pid][s][a[pid]] += 1
                        else:
                            self.dist_id[pid][s][a[pid]] = 0

    def calculate_similarity(self, a, s, next_s, r, pid, delta, i, vis_iters, vis_pids, run, debug_run):
        """
        Calculates the similarity value
        """
        if next_s == 'terminal':
            self.sim_metric = 1  # you can't have miscoordination, because you completed the game
        elif len(self.return_dist[pid][s][a[pid]]) == self.n_samples and delta < 0:
            # delta < 0 because only needed when doing negative updates and calculate only when complete dist
            dist = self.return_dist[pid][s][a[pid]]  # current distribution
            max_action = self.q_values[pid][next_s].index(max(self.q_values[pid][next_s]))
            next_dist = r + self.gamma * np.asarray(self.return_dist[pid][next_s][max_action])

            # Calculate the similarity metric
            if len(next_dist) == 0:
                self.sim_metric = 0
            elif self.metric_name == 'ovl':
                self.sim_metric = ovl(dist, next_dist, self.bins)
            elif self.metric_name == 'ks':
                self.sim_metric = ks(dist, next_dist)
            elif self.metric_name == 'tdl':
                tau = np.array(range(1, len(dist) + 1)) / len(dist)
                self.sim_metric = tdl(dist, next_dist, tau)
            elif self.metric_name == 'hellinger':
                self.sim_metric = hellinger(dist, next_dist, self.bins)
            elif self.metric_name == 'jsd':
                self.sim_metric = jsd(dist, next_dist, self.bins)
            elif self.metric_name == 'emd':
                self.sim_metric = emd(dist, next_dist, self.bins)
            else:
                print('Choose a valid similarity metric')
                self.sim_metric = None
            # Check if I need to visualize
            if i in vis_iters and run == debug_run:
                vis_pid = vis_pids[vis_iters.index(i)]
                if vis_pid == pid:
                    self.plot_histogram(dist, next_dist, pid, a, i)
        else:
            self.sim_metric = 0  # if sim metric not calculated it is 0

    def calculate_alpha_sim(self):
        """
        Calculates correct alpha depending on the algorithm used
        """
        if self.algo in ['ll', 'lsl', 'lsdl']:
            return self.alpha
        elif self.algo in ['lhl', 'hl']:
            return 0
        elif self.algo in ['lhsl', 'hsl']:
            return self.alpha * self.sim_metric
        else:
            print("Use a valid algorithm: ll, lsl, lhl, lhsl, hl, hsl or lsdl")
            exit()

    def calculate_prob(self, a, s, pid):
        """
        Calculates the probability which can be the similarity metric or something else depending on the algorithm used
        """
        if self.algo in ['ll', 'lhl', 'lhsl']:
            return 1 - np.exp(-1 / (self.theta * self.t_values[pid][s][a[pid]]))
        elif self.algo in ['lsl']:
            return self.sim_metric
        elif self.algo in ['hl', 'hsl']:
            return 1
        elif self.algo in ['lsdl']:
            temp_prob = 1 - np.exp(-1 / (self.theta * self.t_values[pid][s][a[pid]]))
            return min(self.sim_metric, temp_prob)
        else:
            print("Use a valid algorithm: ll, lsl, lhl, lhsl, hl, hsl or lsdl")
            exit()

    def collecting_similarity_values_for_plot(self, pid, actions, iteration):
        """
        Collecting the similarity values for the similarity value plot
        """
        if actions == (0, 0):
            self.j_a_0_0[pid][0].append(self.sim_metric)
            self.j_a_0_0[pid][1].append(iteration)
        elif actions == (0, 1):
            self.j_a_0_1[pid][0].append(self.sim_metric)
            self.j_a_0_1[pid][1].append(iteration)
        elif actions == (0, 2):
            self.j_a_0_2[pid][0].append(self.sim_metric)
            self.j_a_0_2[pid][1].append(iteration)
        elif actions == (1, 0):
            self.j_a_1_0[pid][0].append(self.sim_metric)
            self.j_a_1_0[pid][1].append(iteration)
        elif actions == (1, 1):
            self.j_a_1_1[pid][0].append(self.sim_metric)
            self.j_a_1_1[pid][1].append(iteration)
        elif actions == (1, 2):
            self.j_a_1_2[pid][0].append(self.sim_metric)
            self.j_a_1_2[pid][1].append(iteration)
        elif actions == (2, 0):
            self.j_a_2_0[pid][0].append(self.sim_metric)
            self.j_a_2_0[pid][1].append(iteration)
        elif actions == (2, 1):
            self.j_a_2_1[pid][0].append(self.sim_metric)
            self.j_a_2_1[pid][1].append(iteration)
        elif actions == (2, 2):
            self.j_a_2_2[pid][0].append(self.sim_metric)
            self.j_a_2_2[pid][1].append(iteration)

    def plot_histogram(self, cur_dist, next_dist, pid, a, i):
        """
        Plot the histogram for the specified current and target distribution for a particular agent
        """
        action_map = {0: 'A', 1: 'B', 2: 'C'}
        if pid == 0:
            action = action_map[a[0]]
        else:
            action = action_map[a[1]]
        actions_map = {(0, 0): '(A, A)', (0, 1): '(A, B)', (0, 2): '(A, C)', (1, 0): '(B, A)', (1, 1): '(B, B)',
                       (1, 2): '(B, C)', (2, 0): '(C, A)', (2, 1): '(C, B)', (2, 2): '(C, C)'}
        a_joint = actions_map[a]
        plt.figure()
        plt.rcParams.update({'font.size': 13.7})
        plt.subplots_adjust(left=0.15, bottom=0.11, right=0.9, top=0.87, wspace=0.22, hspace=0.35)
        plt.hist(cur_dist, bins=self.bins, label=f'current distribution of action {action}', alpha=0.9)
        plt.hist(next_dist, bins=self.bins, label=f"target distribution after joint-action {a_joint}"
                 , alpha=0.6)
        plt.xlabel("Q-value")
        plt.ylabel("Number of samples")

        plt.title(f"Current and target distribution\n"
                  f"Iteration {i}, agent {pid+1}, $l$ = {round(self.sim_metric, 2)}")
        plt.legend()


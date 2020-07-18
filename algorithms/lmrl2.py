from __future__ import absolute_import
from __future__ import division

import math
from algorithms.similarity_metrics import *


class LMRL2(object):

    def __init__(self, states, beta, leniency, e_decay, t_decay,
                 min_r, max_r, game, algo, metric='dif_hist', init=0):
        # parameters
        self.num_pids = 2  # player ids
        self.num_a = 3
        self.num_states = states
        self.alpha = 0.1
        self.beta = beta
        self.gamma = 1
        self.delta_rec = None
        self.epsilon = None
        self.epsilon_decay = e_decay  # (Important parameter) (with 0.9995 and 0.9999 get later the problem)
        self.min_r = int(min_r)
        self.max_r = int(max_r)
        self.q_values = None
        self.game = game
        self.init = init  # 0 or min_r
        self.algo = algo

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
        # if dist:
        self.return_dist = None
        self.dist_id = None
        self.n_samples = 20
        self.sim_metric = None
        self.metric_name = metric
        if 's' in algo:
            self.dist = True
        else:
            self.dist = False
        self.bins = np.linspace(self.min_r, self.max_r + 0.000001, int(self.max_r - self.min_r) + 1)

    def reset_values(self):
        self.q_values = [[[self.init for _ in range(self.num_a)]
                          for _ in range(self.num_states)] for _ in range(self.num_pids)]
        # maybe change back to 1 if folding doesn't work for self.num_states
        self.t_values = [[[self.MaxTemp for _ in range(self.num_a)]
                          for _ in range(self.num_states)] for _ in range(self.num_pids)]

        self.epsilon = 1
        self.delta_rec = 0
        self.return_dist = [[[[] for _ in range(self.num_a)] for _ in range(self.num_states)]
                            for _ in range(self.num_pids)]
        self.dist_id = [[[0 for _ in range(self.num_a)] for _ in range(self.num_states)]
                        for _ in range(self.num_pids)]
        self.sim_metric = 0

    def select_action(self, pos_a, state, pid_id):
        """Selects an action for a specific pid.
           input: pid number (0 or 1)
           output: action (integer)"""
        # Epsilon-greedy
        rand = random.random()
        if rand < self.epsilon:
            action = random.choice(pos_a)
        else:
            action = self.q_values[pid_id][state].index(max(self.q_values[pid_id][state][0:len(pos_a)]))
        return action

    def next_step(self, a, s, next_s, r):
        self.epsilon = self.epsilon * self.epsilon_decay  # 0.999
        for pid in range(self.num_pids):
            # Calculating delta
            if next_s == 'terminal' or max(self.q_values[pid][next_s]) == math.inf:
                delta = r - self.q_values[pid][s][a[pid]]
            else:
                delta = r + self.gamma * max(self.q_values[pid][next_s]) - self.q_values[pid][s][a[pid]]
            self.delta_rec = delta

            # Calculating the similarity metric
            if self.dist:
                self.calculate_similarity(a, s, next_s, r, pid, delta)
            else:
                self.sim_metric = None

            """ the code below can indicate seven different algorithms
                lenient learning:            (ll)   if alpha_sim = alpha      and prob = temp prob
                lenient similarity learning: (lsl)  if alpha_sim = alpha      and prob = l
                lenient hysteretic learning: (lhl)  if alpha_sim = 0          and prob = temp prob
                lenient hyst. sim. learning: (lhsl) if alpha_sim = alpha * l  and prob = temp prob
                hysteretic learning:         (hl)   if alpha_sim = 0          and prob = 1
                hysteretic sim. learning:    (hsl)  if alpha_sim = alpha * l  and prob = 1
                lenient sim. Daan learning:  (lsdl) if alpha_sim = alpha      and prob = min(l, temp prob)"""

            alpha_sim = self.calculate_alpha_sim()
            prob = self.calculate_prob(a, s, pid)
            # print(f'sim metric: {self.sim_metric}')
            # print(f'alpha: {alpha_sim}')
            # print(f'prob: {prob}')

            # The Q-learning update
            rand = random.random()
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
            if next_s != 'terminal':  # Only updating the temperature value of the first state
                # self.t_values[pid][0][a[pid]] = self.t_decay * self.t_values[pid][0][a[pid]]
                self.t_values[pid][s][a[pid]] = self.t_decay * ((1 - self.temp_diffusion)*self.t_values[pid][s][a[pid]]
                                              + self.temp_diffusion*np.mean(self.t_values[pid][next_s][0:1]))
                # THIS [0:1] IS ONLY FOR THE EXTENDED CLIMB GAME AS AN EASY FIX !
            else:
                # pass
                self.t_values[pid][s][a[pid]] = self.t_decay * self.t_values[pid][s][a[pid]]
            # print(self.t_values[pid][0])
            # Building up the list
            if self.dist:
                prob2 = 1
                if delta >= 0 or rand < prob:
                    if next_s != 'terminal':
                        ret = r + self.gamma * max(self.q_values[pid][next_s])
                    else:
                        ret = r
                    if len(self.return_dist[pid][s][a[pid]]) < self.n_samples:
                        self.return_dist[pid][s][a[pid]].append(ret)
                    # List is complete and replace the oldest value
                    else:
                        self.return_dist[pid][s][a[pid]][self.dist_id[pid][s][a[pid]]] = ret
                        if self.dist_id[pid][s][a[pid]] < (self.n_samples - 1):
                            self.dist_id[pid][s][a[pid]] += 1
                        else:
                            self.dist_id[pid][s][a[pid]] = 0

                    # if next_s != 'terminal' and 2000 < i < 3000:
                    #     cur_dist = self.return_dist[pid][s][a[pid]]
                    #     max_action = self.q_values[pid][next_s].index(max(self.q_values[pid][next_s]))
                    #     next_dist = r + self.gamma * np.asarray(self.return_dist[pid][next_s][max_action])
                    #     # print(f'i: {i}, pid {pid}, l: {self.sim_metric}, actions: {a}, r: {r}, '
                    #     #       f'{cur_dist}, {list(next_dist)}')

    def calculate_similarity(self, a, s, next_s, r, pid, delta):
        if next_s == 'terminal':
            self.sim_metric = 1  # you can't have miscoordination, because you completed the game
            # print(f'hist: {self.sim_metric}')
        elif len(self.return_dist[pid][s][a[pid]]) == self.n_samples and delta < 0:
            # delta < 0 because only needed when doing negative updates and calculate only when complete dist
            dist = self.return_dist[pid][s][a[pid]]  # current distribution
            max_action = self.q_values[pid][next_s].index(max(self.q_values[pid][next_s]))
            next_dist = r + self.gamma * np.asarray(self.return_dist[pid][next_s][max_action])
            if len(next_dist) == 0:
                self.sim_metric = 0
            elif self.metric_name == 'dif_hist':
                self.sim_metric = dif_hist(dist, next_dist, self.bins)
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
            # print(f'hist1: {self.dif_hist(dist, next_dist)}, hist2: {self.dif_hist2(dist, next_dist)}, cur dist: {dist}'
            #       f', next dist: {next_dist}, bins: {self.bins}')
        else:
            self.sim_metric = 0  # if sim metric not calculated it is 0

    def calculate_alpha_sim(self):
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

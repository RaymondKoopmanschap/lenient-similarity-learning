from __future__ import absolute_import
from __future__ import division

import random
# from open_spiel.python.DMARL.helper_functions import select_action

import numpy as np


class Hysteretic(object):

    def __init__(self, num_states, beta, e_decay):
        # parameters
        self.num_states = num_states
        self.temperature = 5000
        self.num_agents = 2
        self.num_actions = 3
        self.alpha = 0.1
        self.beta = beta
        self.omega = 1
        self.delta_rec = 0
        self.gamma = 1
        self.e_decay = e_decay
        self.q_values = [[[0 for i in range(self.num_actions)] for j in range(self.num_states)]
                         for k in range(self.num_agents)]
        self.epsilon = 1

    def reset_values(self):
        self.q_values = [[[0 for i in range(self.num_actions)] for j in range(self.num_states)]
                         for k in range(self.num_agents)]
        self.temperature = 5000
        self.delta_rec = 0
        self.epsilon = 1

    def select_action(self, pos_actions, state, agent_id):
        """Selects an action for a specific agent.
           input: agent number (0 or 1)
           output: action (integer)"""
        # Epsilon-greedy
        rand = random.random()
        if rand < self.epsilon:
            action = random.choice(pos_actions)
        else:
            action = self.q_values[agent_id][state].index(max(self.q_values[agent_id][state][0:len(pos_actions)]))
        self.epsilon = self.epsilon * self.e_decay
        return action

    def next_step(self, actions, cur_state, next_state, reward, i):
        for agent in range(self.num_agents):
            if next_state == 'terminal':
                delta = reward - self.q_values[agent][cur_state][actions[agent]]  # The next state Q-value is 0
            else:
                delta = reward + self.gamma * max(self.q_values[agent][next_state]) \
                               - self.q_values[agent][cur_state][actions[agent]]
            self.delta_rec = delta
            if delta >= 0:
                self.q_values[agent][cur_state][actions[agent]] = self.q_values[agent][cur_state][actions[agent]] \
                                                              + self.alpha * delta
            else:
                self.q_values[agent][cur_state][actions[agent]] = self.q_values[agent][cur_state][actions[agent]] \
                                                              + self.beta * delta


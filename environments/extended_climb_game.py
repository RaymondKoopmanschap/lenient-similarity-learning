import random
import numpy as np


class ExtendedClimbGame(object):
    def __init__(self, game_type='det'):
        self.num_agents = 2
        self.num_states = 10
        self.num_max_actions = 3
        self.terminal = False
        self.game_type = game_type  # det, ps, fs, normal
        self.matrix = [[11, -30, 0], [-30, 7, 6], [0, 0, 5]]
        self.state_action_map = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 0),
                                 5: (1, 1), 6: (1, 2), 7: (2, 0), 8: (2, 1), 9: (2, 2)}

    def new_game(self):
        self.terminal = False
        return 0

    def next_step(self, state, actions):
        """returns state and reward"""
        if 0 < state <= 9:
            a_1, a_2 = self.state_action_map[state]
            self.terminal = True
            state = 'terminal'
            if self.game_type == 'det':
                return state, self.matrix[a_1][a_2]
            elif self.game_type == 'ps':
                if (a_1, a_2) == (1, 1):
                    return state, random.choice([14, 0])
                # if (a_1, a_2) == (0, 2):
                #     return state, np.random.choice([6, 80], p=[0.95, 0.05])
                # if (a_1, a_2) == (0, 0):
                #     return state, random.choice([11, 11])
                else:
                    return state, self.matrix[a_1][a_2]
            elif self.game_type == 'ps2':
                if (a_1, a_2) == (1, 1):
                    return state, random.choice([14, 0])
                if (a_1, a_2) == (2, 2):
                    return state, np.random.choice([16, 0])
                # if (a_1, a_2) == (0, 0):
                #     return state, random.choice([11, 11])
                else:
                    return state, self.matrix[a_1][a_2]
            elif self.game_type == 'fs':
                if (a_1, a_2) == (0, 0):
                    return state, random.choice([10, 12])
                if (a_1, a_2) == (0, 1):
                    return state, random.choice([5, -65])
                if (a_1, a_2) == (0, 2):
                    return state, random.choice([8, -8])
                if (a_1, a_2) == (1, 0):
                    return state, random.choice([5, -65])
                if (a_1, a_2) == (1, 1):
                    return state, random.choice([14, 0])
                if (a_1, a_2) == (1, 2):
                    return state, random.choice([12, 0])
                if (a_1, a_2) == (2, 0):
                    return state, random.choice([5, -5])
                if (a_1, a_2) == (2, 1):
                    return state, random.choice([5, -5])
                if (a_1, a_2) == (2, 2):
                    return state, random.choice([10, 0])
            elif self.game_type == 'normal':
                scale = 5
                if (a_1, a_2) == (0, 2):
                    loc = 10
                    scale = 1
                # if (a_1, a_2) == (1, 1):
                #     loc = random.choice([14, 0])
                #     scale = 10
                # elif (a_1, a_2) == (0, 1):
                #     loc = random.choice([2, 4])
                #     scale = 10
                # elif (a_1, a_2) == (1, 0):
                #     loc = random.choice([2, 4])
                #     scale = 10
                # elif (a_1, a_2) == (1, 2):
                #     loc = random.choice([12, 0])
                else:
                    loc = self.matrix[a_1][a_2]
                return state, np.random.normal(loc=loc, scale=scale)
            else:
                print("invalid type")
            return state
        elif state == 0:
            a_1, a_2 = actions
            if (a_1, a_2) == (0, 0):
                return 1, 0
            if (a_1, a_2) == (0, 1):
                return 2, 0
            if (a_1, a_2) == (0, 2):
                return 3, 0
            if (a_1, a_2) == (1, 0):
                return 4, 0
            if (a_1, a_2) == (1, 1):
                return 5, 0
            if (a_1, a_2) == (1, 2):
                return 6, 0
            if (a_1, a_2) == (2, 0):
                return 7, 0
            if (a_1, a_2) == (2, 1):
                return 8, 0
            if (a_1, a_2) == (2, 2):
                return 9, 0
        else:
            print("invalid state")
            return None

    @staticmethod
    def possible_actions(state):
        if state == 0:
            return [0, 1, 2]
        elif 0 < state <= 9:
            return [0]
        else:
            print('invalid state')

    def get_num_states(self):
        return self.num_states

    def get_max_num_actions(self):
        return self.num_max_actions

    def get_num_agents(self):
        return self.num_agents

    def get_num_actions(self, state):
        if state == 0:
            return self.num_max_actions  # Because only one state
        elif 0 < state <= 9:
            return 1
        else:
            print('invalid state')

    def is_terminal(self):
        return self.terminal

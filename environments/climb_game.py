import random

_NUM_PLAYERS = 2


class ClimbGame(object):
    def __init__(self, game_type='det'):
        self.num_agents = 2
        self.num_states = 1
        self.num_max_actions = 3
        self.terminal = False
        self.game_type = game_type  # det, ps, fs
        self.matrix = [[11, -30, 0], [-30, 7, 6], [0, 0, 5]]

    def new_game(self):
        self.terminal = False
        return 0

    def next_step(self, state, actions):
        """returns state and reward"""
        if state == 0:
            self.terminal = True  # it always reaches the terminal state, since only one state
            state = 'terminal'
        else:
            print("invalid state")
            return None
        a_1, a_2 = actions
        if self.game_type == 'det':
            return state, self.matrix[a_1][a_2]
        if self.game_type == 'ps':
            if actions == (1, 1):
                return state, random.choice([14, 0])
            else:
                return state, self.matrix[a_1][a_2]
        if self.game_type == 'fs':
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
        else:
            print("invalid type")

    @staticmethod
    def possible_actions(state):
        if state == 0:
            return [0, 1, 2]
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

    def is_terminal(self):
        return self.terminal

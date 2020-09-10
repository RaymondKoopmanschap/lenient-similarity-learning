import random


class ExtendedStochasticClimbGame(object):
    """
    The extended stochastic Climb Game as used in my thesis introduced in section 5.3.4, figure 23.
    There is only the partially stochastic (ps) version
    """
    def __init__(self, bc=6, cb=0):
        self.num_agents = 2
        self.num_states = 11
        self.num_max_actions = 3
        self.terminal = False
        self.matrix = [[11, -30, 0], [-30, 7, bc], [0, cb, 5]]  # bc=0 or cb=3 for modified versions
        self.state_action_map = {1: (0, 0), 2: (0, 1), 3: (0, 2), 4: (1, 0),
                                 5: (1, 1), 6: (1, 2), 7: (2, 0), 8: (2, 1), 9: (2, 2), 10: (1, 1)}

    def new_game(self):
        self.terminal = False
        return 0

    def next_step(self, state, actions):
        """returns state and reward"""
        if 0 < state <= 10:
            a_1, a_2 = self.state_action_map[state]
            self.terminal = True
            if (a_1, a_2) == (1, 1):
                if state == 5:
                    return 'terminal', 0
                elif state == 10:
                    return 'terminal', 14
                else:
                    print(f"Something weird happened, state: '{state}' shouldn't be possible")
            else:
                return 'terminal', self.matrix[a_1][a_2]
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
                return self.state_transitions(0.5), 0
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
        elif 0 < state <= 10:
            return [0]
        else:
            print('invalid state')

    @staticmethod
    def state_transitions(prob):
        rand = random.random()
        if rand < prob:
            return 5
        else:
            return 10

    def get_num_states(self):
        return self.num_states

    def get_max_num_actions(self):
        return self.num_max_actions

    def get_num_agents(self):
        return self.num_agents

    def get_num_actions(self, state):
        if state == 0:
            return self.num_max_actions  # Because only one state
        elif 0 < state <= 10:
            return 1
        else:
            print('invalid state')

    def is_terminal(self):
        return self.terminal

import random


class R03Game(object):
    def __init__(self):
        self.num_states = 3
        self.num_max_actions = 3
        self.num_agents = 2
        self.terminal = False

    def new_game(self):
        self.terminal = False
        return 0

    def next_step(self, cur_state, actions):
        reward = 0
        # Terminal state 3
        if cur_state == 'terminal':
            print("Terminal state reached, start new game")
            return cur_state, reward
        # Begin state 0
        elif cur_state == 0:
            if actions == (0, 0):
                state = self.state_transitions(0.5)
            elif actions == (1, 1):
                state = self.state_transitions(0.3)
            elif actions == (1, 2):
                state = self.state_transitions(0.3)
            elif actions == (2, 1):
                state = self.state_transitions(0.3)
            elif actions == (2, 2):
                state = self.state_transitions(0.4)
            else:
                state = 2
            return state, reward
        # State 1
        elif cur_state == 1:
            state = 'terminal'
            if actions == (0, 0) or actions == (1, 1):
                reward = 10
                self.terminal = True
                return state, reward
            elif actions == (0, 1) or actions == (1, 0):
                self.terminal = True
                reward = 0
                return state, reward
        # State 2
        elif cur_state == 2:
            self.terminal = True
            state = 'terminal'
            reward = 0
            return state, reward
        else:
            print("No valid state")

    @staticmethod
    def possible_actions(state):
        if state == 0:
            return [0, 1, 2]
        elif state == 1 or state == 2:
            return [0, 1]
        else:
            print("No available actions anymore")

    @staticmethod
    def state_transitions(prob):
        rand = random.random()
        if rand < prob:
            return 1
        else:
            return 2

    def get_num_states(self):
        return self.num_states

    def get_max_num_actions(self):
        return self.num_max_actions

    def get_num_agents(self):
        return self.num_agents

    def is_terminal(self):
        return self.terminal

    @staticmethod
    def get_num_actions(state):
        if state == 4:
            return None
        elif state == 0:
            return 3
        elif state == 1 or state == 2:
            return 2
        else:
            print("No valid state")

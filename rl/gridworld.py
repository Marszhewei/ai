import numpy as np


class BaseGridworld:
    def __init__(self, width, height, start_state=None, goal_state=None, terminal_states=[]):
        self.state = None
        self.width = width
        self.height = height
        self.start_state = start_state
        self.goal_state = goal_state
        self.terminal_states = terminal_states

        self.reset_state()

    #  四个方向
    def get_possible_actions(self, _):
        all_actions = [(0, 1), (-1, 0), (0, -1), (1, 0)]
        return all_actions

    # 获取状态
    def get_states(self):
        return [(x, y) for x in range(self.width) for y in range(self.height)]

    # 获取奖励，在子类中实现
    def get_reward(self, state, action, next_state):
        raise NotImplementedError

    def get_state_reward_transition(self, state, action):
        next_state = np.array(state) + np.array(action)
        next_state = self._clip_state_to_grid(next_state)
        if self.is_blocked(next_state):
            next_state = state

        next_state = int(next_state[0]), int(next_state[1])
        reward = self.get_reward(state, action, next_state)

        return next_state, reward

    def _clip_state_to_grid(self, state):
        x, y = state
        return np.clip(x, 0, self.width - 1), np.clip(y, 0, self.height - 1)

    # 判断是否到终点
    def is_goal(self, state):
        return tuple(state) == self.goal_state

    # 判断是否到悬崖
    def is_terminal(self, state):
        return tuple(state) in self.terminal_states

    # 重置状态
    def reset_state(self):
        self.state = self.start_state
        return self.state


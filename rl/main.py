from abc import ABC

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import deque

from gridworld import BaseGridworld
from agents import QLearningAgent


# 打印运动策略
def print_optimal_policy(mdp, agent, f=None):
    agent.epsilon = 0
    states_visited, _, _ = agent.run_episode()

    grid = [[' ' for _ in range(mdp.width)] for _ in range(mdp.height)]
    for (x, y) in states_visited:
        grid[y][x] = 'o'

    x, y = mdp.start_state
    grid[y][x] = 'S'
    x, y = mdp.goal_state
    grid[y][x] = 'G'
    for (x, y) in mdp.terminal_states:
        grid[y][x] = 'T'

    grid = grid[::-1]

    print(tabulate(grid, tablefmt='grid'), file=f)
    return grid


# 打印奖励值
def print_values(mdp, agent, f=None):
    grid = [[' ' for _ in range(mdp.width)] for _ in range(mdp.height)]
    for (x, y) in mdp.get_states():
        grid[y][x] = agent.get_value((x, y))

    grid = grid[::-1]

    print(tabulate(grid, tablefmt='grid', floatfmt='.2f'), file=f)
    return grid


# MDP
class CliffGridworld(BaseGridworld, ABC):
    def __init__(self, width, height, start_state, goal_state, terminal_states):
        super().__init__(width, height, start_state, goal_state, terminal_states)

    def get_state_reward_transition(self, state, action):
        next_state = np.array(state) + np.array(action)
        next_state = self._clip_state_to_grid(next_state)
        next_state = int(next_state[0]), int(next_state[1])

        if next_state in self.terminal_states:  # 悬崖
            next_state = self.start_state
            reward = -100
        elif next_state in [(5, 2), (6, 2), (7, 2)]:  # 中间三个圆圈
            reward = -1
        else:  # 正常区域
            reward = -5

        return next_state, reward


def main():
    # 地图长宽
    width, height = 12, 4
    # 起点，终点
    start_state, goal_state = (0, 0), (11, 0)
    # 悬崖
    terminal_states = [(x, 0) for x in range(1, 11)]
    mdp = CliffGridworld(width, height, start_state, goal_state, terminal_states)

    # 奖励总和
    qlearning_sum_rewards = []
    rewards_history = deque(maxlen=10)

    n_episodes = 500

    # Q learning
    agent = QLearningAgent(mdp=mdp)
    for i in range(n_episodes):
        states, actions, rewards = agent.run_episode()
        rewards_history.append(rewards)
        qlearning_sum_rewards.append(np.mean(rewards_history))

    print_optimal_policy(mdp, agent)
    print_values(mdp, agent)

    # 奖励随episode的变化
    plt.plot(np.arange(n_episodes), qlearning_sum_rewards, label='Q-learning')
    plt.ylim(-100, 0)
    plt.xlim(0, 500)
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.legend()

    plt.savefig('figures/res.png')
    plt.close()


if __name__ == '__main__':
    np.random.seed(1)
    main()

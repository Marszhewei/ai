import numpy as np
from collections import defaultdict


class BaseAgent:
    def __init__(self, mdp, run_episode_fn, discount=1, epsilon=0.1, alpha=0.5):
        self.num_updates = None
        self.q_values = None
        self.mdp = mdp
        self.run_episode = lambda: run_episode_fn(mdp, self)
        self.discount = discount
        self.epsilon = epsilon
        self.alpha = alpha

        self.reset()

    def get_action(self, state):
        rand = np.random.rand()
        actions = self.mdp.get_possible_actions(state)
        if rand < self.epsilon:
            return actions[np.random.choice(len(actions))]
        else:
            return self.compute_best_action(state)

    def get_q_value(self, state, action):
        return self.q_values[(state, action)]

    def get_value(self, state):
        return self.compute_value(state)

    def compute_best_action(self, state):
        legal_actions = self.mdp.get_possible_actions(state)
        if legal_actions[0] is None:
            return None
        q_values = [self.get_q_value(state, a) for a in legal_actions]
        eligible_best_actions = [a for i, a in enumerate(legal_actions) if
                                 np.round(q_values[i], 8) == np.round(np.max(q_values), 8)]
        best_action_idx = np.random.choice(len(eligible_best_actions))
        best_action = eligible_best_actions[best_action_idx]
        return best_action

    def compute_q_value(self, state, action):
        next_state, reward = self.mdp.get_state_reward_transition(state, action)
        return reward + self.discount * self.get_value(next_state)

    def compute_value(self, state):
        best_action = self.compute_best_action(state)
        if best_action is None:
            return 0
        else:
            return self.get_q_value(state, best_action)

    def update(self, state, action, reward, next_state):
        raise NotImplementedError

    def reset(self):
        self.q_values = defaultdict(float)
        self.num_updates = 0


# Q-learning
def run_qlearning_episode(mdp, agent):
    states_visited = []
    actions_performed = []
    episode_rewards = 0

    state = mdp.reset_state()
    states_visited.append(state)

    while not mdp.is_goal(state):
        action = agent.get_action(state)
        next_state, reward = mdp.get_state_reward_transition(state, action)

        agent.update(state, action, reward, next_state)
        state = next_state

        states_visited.append(state)
        actions_performed.append(action)
        episode_rewards += reward

    return states_visited, actions_performed, episode_rewards


class QLearningAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(run_episode_fn=run_qlearning_episode, **kwargs)

    def update(self, state, action, reward, next_state):
        q_t0 = self.get_q_value(state, action)
        q_t1 = self.get_value(next_state)

        new_value = q_t0 + self.alpha * (reward + self.discount * q_t1 - q_t0)
        self.q_values[(state, action)] = new_value
        self.num_updates += 1

        return new_value

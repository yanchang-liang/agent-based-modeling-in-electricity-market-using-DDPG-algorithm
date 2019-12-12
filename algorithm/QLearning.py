import numpy as np
import pandas as pd


class QLearningAgents:
    def __init__(self, n_agents, action_space, gamma=0.0):
        self.gamma = gamma
        self.n_agents = n_agents
        self.agents = [QLearningTable(action_space, gamma=self.gamma) for _ in range(self.n_agents)]

    def select_action(self, obs):
        action = [self.agents[i].choose_action(obs) for i in range(self.n_agents)]
        return np.array(action)

    def learn(self, s, a, r, s_):
        if self.n_agents == 1:
            self.agents[0].learn(s, a, r, s_)
        else:
            for i in range(self.n_agents):
                self.agents[i].learn(s, a[i], r[i], s_)


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, gamma=0.0, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = e_greedy
        self.n_steps = 0
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        self.n_steps += 1
        if self.epsilon > 0.1:
            self.epsilon = 0.9993 ** self.n_steps
        else:
            self.epsilon = 0.05

        if np.random.uniform() > self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

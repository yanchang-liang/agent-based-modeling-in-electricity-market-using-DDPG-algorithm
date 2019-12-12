import numpy as np


class RothErevAgents:
    def __init__(self, n_agents, action_space):
        self.n_agents = n_agents
        self.agents = [RothErev(action_space) for _ in range(self.n_agents)]

    def select_action(self):
        action = [self.agents[i].choose_action() for i in range(self.n_agents)]
        return np.array(action)

    def learn(self, reward, r=0.1, e=0.2):
        for i in range(self.n_agents):
            self.agents[i].learn(reward[i], r=r, e=e)


class RothErev:
    def __init__(self, action_space):
        self.action_space = action_space
        self.n_strategies = len(action_space)
        self.strategy_value = np.ones(self.n_strategies) * 1000
        self.latest_ind = None

    def choose_action(self):
        k = 2.0
        c = k / self.n_strategies * self.strategy_value.sum()
        exp_strategy_value = np.exp(self.strategy_value / c)
        sum_exp_strategy_value = exp_strategy_value.sum()
        strategy_prob = exp_strategy_value / sum_exp_strategy_value
        cumsum_strategy_prob = strategy_prob.cumsum(0)
        random_number = np.random.rand()
        for ind in range(self.n_strategies):
            if random_number <= cumsum_strategy_prob[ind]:
                self.latest_ind = ind

                return self.action_space[ind]

        return self.action_space[self.latest_ind]

    def learn(self, reward, r=0.1, e=0.9):
        for ind in range(self.n_strategies):
            if ind == self.latest_ind:
                self.strategy_value[ind] = (1 - r) * self.strategy_value[ind] + (1 - e) * reward
            else:
                self.strategy_value[ind] = (1 - r) * self.strategy_value[ind] + \
                                           e * self.strategy_value[ind] / self.n_strategies

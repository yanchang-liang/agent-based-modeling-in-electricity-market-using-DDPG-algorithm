import numpy as np
from market.three_bus import market_clearing
from algorithm.VRE import RothErevAgents
import matplotlib.pyplot as plt

n_agents = 2
action_space = np.arange(0, 3.1, 0.2)
n_steps = 10000
a_real = np.array([15.0, 18.0])
strategic_variables = np.zeros((n_steps, n_agents))

multi_agents = RothErevAgents(n_agents, action_space)

for step in range(n_steps):
    action = multi_agents.select_action()
    alpha = action * a_real
    nodal_price, profit = market_clearing(alpha)

    strategic_variables[step] = alpha

    multi_agents.learn(profit)

    if (step + 1) % 1000 == 0:
        print('Step:', step + 1, 'a1: %.2f' % alpha[0], 'a2: %.2f' % alpha[1],
              'r1: %.3f' % profit[0], 'r2: %.3f' % profit[1])


C = np.array([[0.36, 0.58, 0.75],
              [0.92, 0.28, 0.29]])

plt.plot(strategic_variables[:, 0], lw=0.5, C=C[0], alpha=0.5, label=r"$\alpha_{1t}$")
plt.plot(strategic_variables[:, 1], lw=0.5, C=C[1], alpha=0.5, label=r"$\alpha_{2t}$")
plt.plot([0, 10000], [20.29, 20.29], '--', C=C[0], label=r"$\alpha_{1t}^\ast$")
plt.plot([0, 10000], [22.98, 22.98], '--', C=C[1], label=r"$\alpha_{2t}^\ast$")
plt.xlabel(r"$t$")
plt.ylabel(r"$\alpha_{gt}$ (\$/MHh)")
plt.title("VRE (3-Bus System)")
plt.legend()
plt.savefig('VRE.png', dpi=600)
plt.show()

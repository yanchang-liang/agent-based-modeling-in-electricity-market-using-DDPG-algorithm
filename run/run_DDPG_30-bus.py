import numpy as np
from market.thirty_bus import market_clearing
from algorithm.DDPG import DDPG
from algorithm.model import ANet1, CNet1
import matplotlib.pyplot as plt

n_agents = 6
n_states = 6
n_actions = 1
n_steps = 10000
var = 1

a_real = np.array([18.0, 20.0, 25.0, 22.0, 22.0, 16.0])

a = np.zeros(n_agents)
s_ = np.random.rand(n_agents)
alpha = np.zeros(n_agents)
strategic_variables = np.zeros((n_steps, n_agents))

gencos = []
for _ in range(n_agents):
    gencos.append(DDPG(n_states, n_actions, ANet1, CNet1))

for step in range(n_steps):
    s = s_
    for g in range(n_agents):
        a[g] = gencos[g].choose_action(s)
        a[g] = np.clip(a[g] + np.random.randn(1) * var, -1, 1)

    alpha = (a + 1) * a_real * 1.5  # strategic variable
    nodal_price, profit = market_clearing(alpha)
    strategic_variables[step] = alpha
    r = profit / 1000

    for g in range(n_agents):
        s_ = nodal_price / 30
        gencos[g].store_transition(s, a[g], r[g], s_)

    if 1000 <= step < 9000:
        for g in range(n_agents):
            gencos[g].learn()
        if var > 0.05:
            var *= 0.9993
    elif step >= 9000:
        var = 0

    if (step + 1) % 1000 == 0:
        print('Step:', step + 1, 'a1: %.2f' % alpha[0], 'a2: %.2f' % alpha[1], 'a3: %.2f' % alpha[2],
              'a4: %.2f' % alpha[3], 'a5: %.2f' % alpha[4], 'a6: %.2f' % alpha[5], 'Explore: %.2f' % var)


C = np.array([[0.90, 0.19, 0.20],
              [0.29, 0.54, 0.75],
              [0.37, 0.72, 0.36],
              [1.00, 0.55, 0.10],
              [0.96, 0.89, 0.47],
              [0.69, 0.40, 0.24]])

plt.plot(strategic_variables[:, 0], lw=0.1, C=C[0], alpha=0.5, label=r"$\alpha_{1t}$")
plt.plot(strategic_variables[:, 1], lw=0.1, C=C[1], alpha=0.5, label=r"$\alpha_{2t}$")
plt.plot(strategic_variables[:, 2], lw=0.1, C=C[2], alpha=0.5, label=r"$\alpha_{3t}$")
plt.plot(strategic_variables[:, 3], lw=0.1, C=C[3], alpha=0.5, label=r"$\alpha_{4t}$")
plt.plot(strategic_variables[:, 4], lw=0.1, C=C[4], alpha=0.5, label=r"$\alpha_{5t}$")
plt.plot(strategic_variables[:, 5], lw=0.1, C=C[5], alpha=0.5, label=r"$\alpha_{6t}$")
plt.plot([0, 10000], [21.388, 21.388], '--', C=C[0])
plt.plot([0, 10000], [23.807, 23.807], '--', C=C[1])
plt.plot([0, 10000], [34.317, 34.317], '--', C=C[2])
plt.plot([0, 10000], [27.235, 27.235], '--', C=C[3])
plt.plot([0, 10000], [24.848, 24.848], '--', C=C[5])
plt.xlabel(r"$t$")
plt.ylabel(r"$\alpha_{gt}$ (\$/MHh)")
plt.title("DDPG (IEEE 30-Bus System)")
plt.legend()
plt.savefig('DDPG_30-bus.png', dpi=600)
plt.show()

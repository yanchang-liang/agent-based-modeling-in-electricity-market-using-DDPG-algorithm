import numpy as np
from market.three_bus import market_clearing
from algorithm.DDPG import DDPG
from algorithm.model import ANet2, CNet2
import matplotlib.pyplot as plt

n_agents = 2
n_states = 2
n_actions = 1
n_steps = 10000
var = 1

a_real = np.array([15.0, 18.0])

a = np.zeros(n_agents)
s_ = np.random.rand(n_agents)
alpha = np.zeros(n_agents)
strategic_variables = np.zeros((n_steps, n_agents))

gencos = []
for _ in range(n_agents):
    gencos.append(DDPG(n_states, n_actions, ANet2, CNet2))

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
        if var > 0.03:
            var *= 0.999
    elif step >= 9000:
        var = 0

    if (step + 1) % 1000 == 0:
        print('Step:', step + 1, 'a1: %.2f' % alpha[0], 'a2: %.2f' % alpha[1], 'r1: %.3f' % profit[0],
              'r2: %.3f' % profit[1], 'Explore: %.2f' % var)


C = np.array([[0.36, 0.58, 0.75],
              [0.92, 0.28, 0.29]])

plt.plot(strategic_variables[:, 0], lw=0.1, C=C[0], alpha=0.5, label=r"$\alpha_{1t}$")
plt.plot(strategic_variables[:, 1], lw=0.1, C=C[1], alpha=0.5, label=r"$\alpha_{2t}$")
plt.plot([0, 10000], [20.29, 20.29], '--', C=C[0], label=r"$\alpha_{1t}^\ast$")
plt.plot([0, 10000], [22.98, 22.98], '--', C=C[1], label=r"$\alpha_{2t}^\ast$")
plt.xlabel(r"$t$")
plt.ylabel(r"$\alpha_{gt}$ (\$/MHh)")
plt.title("DDPG (3-Bus System)")
plt.legend()
plt.savefig('DDPG_3_bus.png', dpi=600)
plt.show()

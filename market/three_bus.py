import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False


# 市场出清，考虑网络阻塞
def market_clearing(alpha):

    # 供给曲线的截距和斜率
    a_real = np.array([15.0, 18.0])
    b_real = np.array([0.01, 0.008])

    # 需求曲线的截距和斜率
    c_real = np.array([40.0, 40.0]) * -1
    d_real = np.array([0.08, 0.06])

    # 机组功率上下限
    p_min = np.array([0.0, 0.0])
    p_max = np.array([500.0, 500.0])

    # 负荷需求上下限
    q_min = np.zeros(2)
    q_max = np.array([500.0, 666.666666666667])

    J_g = ([[-0.333333333333333, -0.333333333333333, -0.666666666666667],
            [0.333333333333334, -0.666666666666667, -0.333333333333333],
            [0, 0, 0]])

    J = np.array([[-0.333333333333333, 0.0, 0.333333333333333, -0.333333333333334],
                  [-0.333333333333333, 0.0, 0.333333333333333, 0.666666666666667],
                  [-0.666666666666667, 0.0, 0.666666666666667, 0.333333333333333]])

    J_max = np.array([25.0, 1000.0, 1000.0, 25.0, 1000.0, 1000.0])

    P = matrix(np.diag(np.append(b_real, d_real)))
    q = matrix(np.append(alpha, c_real))
    G = matrix(np.vstack((J, -J, np.diag(-np.ones(4)), np.diag(np.ones(4)))))
    h = matrix(np.hstack((J_max, p_min, q_min, p_max, q_max)))
    A = matrix(np.hstack((-np.ones(2), np.ones(2)))).T
    b = matrix(0.0)

    sv = solvers.qp(P, q, G, h, A, b)

    miu1 = sv['z'][0:3]
    miu2 = sv['z'][3:6]

    nodal_price = (np.ones((3, 1)) * sv['y'][0] - np.dot(J_g, miu1 - miu2)).squeeze()
    nodal_price_g = np.array([nodal_price[0], nodal_price[2]])
    mc_amount = np.array(sv['x'][:2]).squeeze()
    cost_real = 0.5 * b_real * mc_amount ** 2 + a_real * mc_amount
    cost_declare = mc_amount * np.transpose(nodal_price_g)
    profit = cost_declare - cost_real

    return nodal_price_g, profit


if __name__ == '__main__':

    alpha = np.array([20.29, 22.98])
    print(market_clearing(alpha))

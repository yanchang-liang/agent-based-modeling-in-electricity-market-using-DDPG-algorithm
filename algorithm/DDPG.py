import numpy as np
import torch
import torch.nn as nn


class DDPG:
    def __init__(self, s_dim, a_dim, ANet, CNet, memory_capacity=1000, gamma=0.0, lr_a=0.001, lr_c=0.001):
        self.a_dim, self.s_dim = a_dim, s_dim
        self.gamma = gamma
        self.memory_capacity = memory_capacity
        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.Actor_eval = ANet(s_dim, a_dim)
        self.Actor_target = ANet(s_dim, a_dim)
        self.Critic_eval = CNet(s_dim, a_dim)
        self.Critic_target = CNet(s_dim, a_dim)
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=lr_a)
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=lr_c)
        self.loss_td = nn.MSELoss()

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.Actor_eval(s)[0].detach()

    def learn(self, batch_size=100, tau=0.01):
        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1 - tau))')
            eval('self.Actor_target.' + x + '.data.add_(tau*self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1 - tau))')
            eval('self.Critic_target.' + x + '.data.add_(tau*self.Critic_eval.' + x + '.data)')

        indices = np.random.choice(self.memory_capacity, batch_size)
        bt = self.memory[indices, :]
        bs = torch.FloatTensor(bt[:, :self.s_dim])
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])

        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs, a)
        loss_a = -torch.mean(q)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_)
        q_ = self.Critic_target(bs_, a_)
        q_target = br + self.gamma * q_
        q_v = self.Critic_eval(bs, ba)
        td_error = self.loss_td(q_target, q_v)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity
        self.memory[index, :] = transition
        self.pointer += 1

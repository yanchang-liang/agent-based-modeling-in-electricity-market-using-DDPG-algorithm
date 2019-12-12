import torch
import torch.nn as nn
import torch.nn.functional as F


class ANet1(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(ANet1, self).__init__()
        self.FC1 = nn.Linear(s_dim, 128)
        self.FC2 = nn.Linear(128, 64)
        self.FC3 = nn.Linear(64, a_dim)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = torch.tanh(self.FC3(result))
        return result


class CNet1(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CNet1, self).__init__()
        self.FC1 = nn.Linear(s_dim, 128)
        self.FC2 = nn.Linear(128 + a_dim, 128)
        self.FC3 = nn.Linear(128, 64)
        self.FC4 = nn.Linear(64, 1)

    def forward(self, obs, acts):
        result = F.relu(self.FC1(obs))
        combined = torch.cat([result, acts], 1)
        result = F.relu(self.FC2(combined))
        return self.FC4(F.relu(self.FC3(result)))


class ANet2(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(ANet2,self).__init__()
        self.fc1 = nn.Linear(s_dim, 64)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(64, a_dim)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        return x


class CNet2(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CNet2,self).__init__()
        self.fcs = nn.Linear(s_dim, 64)
        self.fcs.weight.data.normal_(0, 0.1)
        self.fca = nn.Linear(a_dim, 64)
        self.fca.weight.data.normal_(0, 0.1)
        self.fcsa = nn.Linear(64, 64)
        self.fcsa.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(64, 1)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        xy = F.relu(x + y)
        xy = F.relu(self.fcsa(xy))
        actions_value = self.out(xy)
        return actions_value

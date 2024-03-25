import torch
from torch import nn
import numpy as np
import random


class SoftClusterNetwork(nn.Module):
    """"""

    def __init__(self, out_shape, w=None, h=None):
        """"""
        super(SoftClusterNetwork, self).__init__()
        self.image_init(out_shape, w, h)
        self.n_abstract_state = out_shape


    def dense_init(self, out_shape, w=None, h=None):
        self.fc1 = nn.Linear(2, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, out_shape)


    def image_init(self, out_shape, w, h):
        self.conv1 = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=1, stride=1, padding=0, dilation=1):
            return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, kernel_size=1), kernel_size=5, padding=1), kernel_size=3, padding=1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, kernel_size=1), kernel_size=5, padding=1), kernel_size=3, padding=1)
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, out_shape)

    def forward(self, obs):
        logits = self.image_forward(obs)
        return logits

    def dense_forward(self, obs):
        obs = torch.flatten(obs, start_dim=1)
        x = torch.selu(self.bn1(self.fc1(obs)))
        x = torch.selu(self.bn2(self.fc2(x)))
        x = torch.selu(self.bn3(self.fc3(x)))
        logits = self.fc4(x)

        return logits

    def image_forward(self, obs):
        obs = obs.permute(0, 3, 1, 2)
        x = torch.selu(self.bn1(self.conv1(obs)))
        x = torch.selu(self.bn2(self.conv2(x)))
        x = torch.selu(self.bn3(self.conv3(x)))
        x = torch.selu(self.bn4(self.head(x.reshape(x.size(0), -1))))
        return self.out(x)

    def prob(self, logits, temperature):
        return torch.softmax(logits/temperature, dim=-1)

    def gumbel_prob(self, logits, temperature):
        return nn.functional.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)

    def pred(self, obs, temperature):
        logits = self.forward(obs)
        return torch.clamp(torch.softmax(logits/temperature, dim=-1), 1e-9, 1 - (self.n_abstract_state * 1e-9))

    def gumbel_pred(self, obs, temperature):
        logits = self.forward(obs)
        return nn.functional.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


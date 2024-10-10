# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:02:21 2023

@author: cai03
"""

import torch
from torch import nn
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.cnn2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5,padding=0, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(400, 120)
        self.f2 = nn.Linear(120, 84)
        self.f3 = nn.Linear(84, 10)

    def forward(self, x): #前向计算
        c1 = self.relu(self.cnn1(x))
        c2 = self.pool1(c1)
        c3 = self.relu(self.cnn2(c2))
        c4 = self.pool2(c3)
        c5 = self.flatten(c4)
        c6 = self.relu(self.f1(c5))
        c7 = self.relu(self.f2(c6))
        output = self.f3(c7)

        return output

    a = LeNet(nn.Module)
    a.f1 = nn.Linear(84, 1)

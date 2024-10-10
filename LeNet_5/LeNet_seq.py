# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:02:21 2023

@author: cai03
"""

from torch import nn
def model_Seq():
    model = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2, stride=1),
                          # 第一层卷积层，输入为（1，6，5）， 填充为2， 步幅为1
                          nn.ReLU(), # 过ReLU层
                          nn.AvgPool2d(kernel_size=(2, 2), stride=2), # 第一层池化层，感受野为（2，2），步幅为2
                          nn.Conv2d(6, 16, kernel_size=5, padding=0, stride=1),
                          # 第二层卷积层，输入为（6，16，5）， 填充为0， 步幅为1
                          nn.ReLU(), # 过ReLU层
                          nn.AvgPool2d(kernel_size=(2, 2), stride=2), # 第二层池化层，感受野为（2，2），步幅为2
                          nn.Flatten(), # 数据展平
                          nn.Linear(16 * 25, 120), # 第一层全连接
                          nn.ReLU(),
                          nn.Linear(120, 84), # 第二层全连接
                          nn.ReLU(),
                          nn.Linear(84, 10)) # 第三层全连接
    return model
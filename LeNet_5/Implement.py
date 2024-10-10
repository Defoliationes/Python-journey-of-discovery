# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:56:45 2023

@author: cai03
"""

from LeNet_Train import Training
from LeNet_class import LeNet
from LeNet_seq import model_Seq
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data

def Data_acquisition():
    """
    data_train训练集
    data_test测试集

    :return: data_train,data_test
    """
    data_train = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True)
    data_test = FashionMNIST(root='./data',
                             train=False,
                             transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                             download=True)
    return data_train, data_test


def data_processing(data_train, data_test):
    """
    :param: data_train：未划分的数据集

    再用DataLoader函数将数分批次

    :return: train_dataloader,val_dataloader
    """
    train_dataloader = Data.DataLoader(dataset=data_train,
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=4)

    test_dataloader = Data.DataLoader(dataset=data_test,
                                      batch_size=64,
                                      shuffle=True,
                                      num_workers=4)
    return train_dataloader, test_dataloader

if __name__ == "__main__":
    LeNet_class = LeNet()
    LeNet_seq = model_Seq()

    data_train, data_test = Data_acquisition()
    train_dataloader, test_dataloader = data_processing(data_train, data_test)

    Training = Training(train_dataloader, test_dataloader)
    train_process = Training.train(LeNet_class, 20, 'class')
    #train_process = Training.train(LeNet_seq, 20, 'sequential')




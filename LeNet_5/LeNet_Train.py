# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 14:38:12 2023

@author: cai03
"""
import copy
import torch

import torch.nn as nn
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Line, Page
import time


class Training:
    def __init__(self, train_dataloader, test_dataloader):
        self.Str = None
        self.train_process = None
        self.model = None
        self.optimizer = None
        self.since = None

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()#包含softmax函数

    def Model_training(self, model, num_epochs):
        """
        :param model: 模型
        :param train_dataloader: 划分过的训练集
        :param test_dataloader: 划分过的验证集
        :param num_epochs: 训练轮次

        设置设备device
        设置优化器 optimizer
        设置损失函数类型criterion
        数据的初始化
        （轮次循环（测试集循环&验证集循环））
        保留最佳参数
        将训练结果放到DataFrame中

        :return:train_process
        """
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 设置优化器Adam

        self.model = model.to(self.device)

        best_model_wts = copy.deepcopy(self.model.state_dict)
        
        best_acc = 0.0
        
        train_loss_all = []
        train_acc_all = []

        test_loss_all = []
        test_acc_all = []
        
        self.since = time.time() # 记录开始时间

        train_process, best_model_wts = self.Rounds(best_acc, train_loss_all, test_loss_all,
                                train_acc_all, test_acc_all, best_model_wts, num_epochs)

        model.load_state_dict(best_model_wts)# 记录参数
        torch.save(model.load_state_dict(best_model_wts), './best_model_{}.pth'.format(self.Str))#将参数保存的pth文件中

        # 将轮次训练完后的数据结果存放到DataFrame中

        return train_process

    def Rounds(self, best_acc, train_loss_all, test_loss_all, train_acc_all,
               test_acc_all, best_model_wts, num_epochs):
        """
        :param best_acc:
        :param train_loss_all:
        :param test_loss_all:
        :param train_acc_all:
        :param test_acc_all:
        :param best_model_wts:
        :param num_epochs:

        （轮次循环）
        损失值的初始化
        正确个数的初始化
        数据总和的初始化
        （训练集训练&验证集训练）
        获取平均损失函数值
        获取精确度

        :return:
        """
        for epoch in range(num_epochs):
            print("#" + "=-" * 5 + "Epoch {} / {} {}".format(epoch + 1, num_epochs, self.Str) + "-=" * 5 + "#")
            train_loss = 0.0
            train_corrects = 0
            test_loss = 0.0
            test_corrects = 0
            train_num = 0
            test_num = 0

            train_loss, train_corrects, train_num = self.train_cycle(train_loss,
                                                                train_corrects, train_num)

            test_loss, test_corrects, test_num = self.eval_cycle(test_loss,
                                                            test_corrects, test_num)

            train_loss_all, train_acc_all = self.app(train_loss_all,
                                                train_acc_all, train_loss,
                                                train_corrects, train_num)

            test_loss_all, test_acc_all = self.app(test_loss_all,
                                              test_acc_all, test_loss,
                                              test_corrects, test_num)

            print("{} Train loss: {:.4f}; Train acc:{:.4f}".format(epoch+1, train_loss_all[-1], train_acc_all[-1]))
            print("{} Test loss: {:.4f}; Test acc:{:.4f}".format(epoch+1, test_loss_all[-1], test_acc_all[-1]))

            # 获取最佳参数
            if test_acc_all[-1] > best_acc:
                best_acc = test_acc_all[-1]  # 找到精确度最高的训练数据
                best_model_wts = copy.deepcopy(self.model.state_dict())  # 令其参数为最佳参数

            time_use = time.time() - self.since  # 计算训练时间
            print("训练和验证耗费时长为：{:.0f}m{:.0f}s\n".format(time_use // 60, time_use % 60))  # 打印

        train_process = pd.DataFrame(data={"epoch": range(num_epochs),
                                           "train_loss_all": train_loss_all,
                                           "test_loss_all": test_loss_all,
                                           "train_acc_all": train_acc_all,
                                           "test_acc_all": test_acc_all})

        self.train_process = train_process
        train_process.to_excel("{}.xlsx".format(self.Str))
        return train_process, best_model_wts

    def train_cycle(self, loss_num, corrects, num):
        """
        :param loss_num:
        :param corrects:
        :param num:

        （训练集训练）
        将数据集放入设备中
        将模型调为训练模式
        数据进行训练，并通过softmax层
        计算单批次损失函数值
        梯度化零，后向传播
        计算损失函数值总和
        计算预测集正确个数
        计算训练数据个数

        :return:
        """
        for step, (b_x, b_y) in enumerate(self.train_dataloader):
            b_x = b_x.to(self.device)
            b_y = b_y.to(self.device)
            self.model.train()

            output = self.model(b_x)

            pre_lab = torch.argmax(output, dim=1)

            loss = self.criterion(output, b_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_num += loss.item() * b_x.size(0)
            corrects += torch.sum(pre_lab == b_y.data)
            num += b_x.size(0)

        return loss_num, corrects, num

    def eval_cycle(self, loss_num, corrects, num):
        """
        :param loss_num:
        :param corrects:
        :param num:

        （测试集训练）
        将数据集放入设备中
        将模型调为训练模式
        数据进行训练，并通过softmax层
        计算单批次损失函数值
        计算损失函数值总和
        计算预测集正确个数
        计算训练个数

        :return:
        """
        for step, (b_x, b_y) in enumerate(self.test_dataloader):

            b_x = b_x.to(self.device)
            b_y = b_y.to(self.device)
            self.model.eval()

            output = self.model(b_x)

            pre_lab = torch.argmax(output, dim=1)

            loss = self.criterion(output, b_y)

            loss_num += loss.item() * b_x.size(0)
            corrects += torch.sum(pre_lab == b_y.data)
            num += b_x.size(0)

        return loss_num, corrects, num

    def app(self, loss_all, acc_all, loss, corrects, num):
        """
        :param loss_all:
        :param acc_all:
        :param loss:
        :param corrects:
        :param num:


        :return:
        """
        loss_all.append(loss / num)
        loss_all = [float('{:.4f}'.format(i)) for i in loss_all]
        acc_all.append(corrects.double().item() / num)
        acc_all = [float('{:.4f}'.format(i)) for i in acc_all]
        return loss_all, acc_all

    def matplot_train_process(self, train_process):
        """
        :param train_process: 训练结果

        整理数据
        运用pyecharts绘制图形
        将图形保存到html文件中

        :return: grid.render("./cnn_Image_{}.html".format(Str))
        """
        x_data = train_process["epoch"]

        y_a_1 = train_process.train_loss_all
        y_b_1 = train_process.test_loss_all

        y_a_2 = train_process.train_acc_all
        y_b_2 = train_process.test_acc_all

        line_1 = (
            Line()
            .add_xaxis(x_data)
            .add_yaxis('train_loss', y_a_1)
            .add_yaxis('test_loss', y_b_1)
            .set_global_opts(title_opts=opts.TitleOpts(title='训练与测试损失值'),
                             yaxis_opts=opts.AxisOpts(name='损失值', name_location='center', name_gap=30))
        )
        line_2 = (
            Line()
            .add_xaxis(x_data)
            .add_yaxis('train_acc', y_a_2)
            .add_yaxis('test_acc', y_b_2)
            .set_global_opts(title_opts=opts.TitleOpts(title='训练与测试精确度'),
                             yaxis_opts=opts.AxisOpts(name='精确度', name_location='center', name_gap=30))
        )

        page = Page()
        page.add(line_1, line_2)
    
        return page.render("./cnn_Image_{}.html".format(self.Str))
    
    def train(self, model, num_epochs, Str):
        """
        :param model: 模型
        :param num_epochs: 轮次总数
        :param Str: 字符标签

        :return:train_process
        """
        self.Str = Str
        train_process = self.Model_training(model, num_epochs)
        self.matplot_train_process(train_process)
        return train_process
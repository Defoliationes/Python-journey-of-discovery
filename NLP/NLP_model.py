"""
定义模型

模型优化方法：
# 添加一个新的全连接层作为输出层，激活函数处理
# 把双向的LSTM的output穿一个单向LSTM再进行处理
# 超参数的修改
"""
import torch.nn as nn
import torch.nn.functional as F
import NLP_lib as lib
import numpy as np
import torch
import os
from NLP_data import get_dataloader
from torch.optim import Adam
from tqdm import tqdm
import matplotlib.pyplot as plt


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.embedding = nn.Embedding(len(lib.ws), 100)
        self.lstm = nn.LSTM(input_size=100, hidden_size=lib.hidden_size,
                num_layers=lib.num_layers, batch_first=True,
                bidirectional=lib.bidriectional,dropout=lib.dropout)
        self.fc = nn.Linear(lib.hidden_size*2, 2)

    def forward(self, input):
        """

        :param input:[batch_size, max_len]
        """
        x = self.embedding(input) # 进行embedding操作，形状：[batch_size, max_len, 100]
        # x: [batch_size, max_len, 2*hidden_size], h_n:[2*2, batch_size, hidden_size]
        x, (h_n, c_n) = self.lstm(x)

        # 获取两个方向最后一次的output， 进行concat操作
        output_fw = h_n[-2, :, :] #正向的最后一次输出
        output_bw = h_n[-1, :, :] #反向的最后一次输出
        output = torch.cat([output_fw, output_bw], dim=-1) # [batch_size, hidden_size*2]

        out = self.fc(output) # 添加一个新的全连接层作为输出层，激活函数处理

        return F.log_softmax(out, dim=-1)



model = MyModel().to(lib.device)
optimizer = Adam(model.parameters(), 0.001)

if os .path.exists("C:/Users/cai03/PycharmProjects/pythonProject/NLP/model.pkl"):
    model.load_state_dict(torch.load("C:/Users/cai03/PycharmProjects/pythonProject/NLP/model.pkl"))
    optimizer.load_state_dict(torch.load("C:/Users/cai03/PycharmProjects/pythonProject/NLP/optimizer.pkl"))


def train(epoch):
    loss_list = []
    data_loader = get_dataloader(train=False, batch_size=lib.test_batch_size)
    for idx, (input, target) in tqdm(enumerate(data_loader), total=len(data_loader), ascii=True, desc="训练："):
        input = input.to(lib.device)
        target = target.to(lib.device)
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output, target)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

        if idx % 100 == 0 :
            torch.save(model.state_dict(), "C:/Users/cai03/PycharmProjects/pythonProject/NLP/model.pkl")
            torch.save(optimizer.state_dict(), "C:/Users/cai03/PycharmProjects/pythonProject/NLP/optimizer.pkl")

    print(epoch, np.mean(loss_list))
    return np.mean(loss_list)

def eval():
    loss_list = []
    acc_list = []
    data_loader = get_dataloader(train=False, batch_size=lib.test_batch_size)
    for idx, (input, target) in tqdm(enumerate(data_loader), total=len(data_loader), ascii=True, desc="测试："):
        input = input.to(lib.device)
        target = target.to(lib.device)
        with torch.no_grad():
            output = model(input)
            cur_loss = F.nll_loss(output, target)
            loss_list.append(cur_loss.cpu().item())
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc.cpu().item())
    print(f"Mean Loss:{np.mean(loss_list)}, Mean Acc:{np.mean(acc_list)}")

def plot_show(y, x):
    plt.plot(x, y, marker='o', linestyle='-', color='b')
    for j, txt in enumerate(y):
        plt.annotate(txt, (x[j], y[j]),
                     textcoords="offset points", xytext=(0, 10), ha='center')
    plt.title('Mean Train Loss')
    # 添加x轴和y轴的标签
    plt.xlabel('Epoch')
    plt.ylabel('mean_train_loss')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    '''
    train_loss = []
    epochs = 20
    for i in range(epochs):
        mean_train_loss = train(i+1)
        train_loss.append(mean_train_loss)
    plot_show(x=np.arange(epochs), y=train_loss)
    '''
    eval()

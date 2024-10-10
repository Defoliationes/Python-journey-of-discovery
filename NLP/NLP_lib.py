import pickle

import torch

ws = pickle.load(open("C:/Users/cai03/PycharmProjects/pythonProject/NLP/ws.pkl","rb"))

max_len = 200
batch_size = 512
test_batch_size = 1000

hidden_size = 128
num_layers = 2
bidriectional = True
dropout = 0.4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
@author:Yangt
@file:tc.py
@time:2022/03/01
@version:
"""
import os

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import torch.utils.data as Data
from torch.autograd import Variable
import torch.nn.functional as F
import jieba

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.metrics import classification_report

def LoadData():
    train_df, dev_df, test_df = pd.read_csv('data/om/train.csv'), pd.read_csv('data/om/dev.csv'), pd.read_csv('data/om/test.csv')
    train, dev, test = np.array(train_df), np.array(dev_df), np.array(test_df)
    # print(train_df)
    # print(dev_df)
    # print('\n')
    # print(test_df)
    data_train = [train[i][2] for i in range(len(train))]
    label_train = [train[i][0] for i in range(len(train))]

    data_dev = [dev[i][2] for i in range(len(dev))]
    label_dev = [dev[i][0] for i in range(len(dev))]

    data_test = [test[i][2] for i in range(len(test))]
    label_test = [test[i][0] for i in range(len(test))]

    return data_train, label_train, data_dev, label_dev, data_test, label_test


def cut(seq, make_vocab=False):
    seq_cut = []
    seq_cut_list = []
    for i in seq:
        cut_res = list(jieba.cut(i))
        seq_cut = seq_cut + cut_res
        # print(seq_cut)
        seq_cut_list.append(cut_res)
    if make_vocab:
        word2num = sorted(collections.Counter(seq_cut).items(), key=lambda item: item[1], reverse=True)
        # print(word2num)
        # 所有词
        vocab = list(set(seq_cut))
        # 词对应索引
        word2index = {w[0]: i+1 for i, w in enumerate(word2num)}
        word2index["<PAD>"] = 0
        word2index["<UNK>"] = len(word2index)
        return seq_cut_list, word2index, vocab
    else:
        return seq_cut_list


def make_data(seq, label):
    inputs = []
    for i in seq:
        # seq_index = [word2index[word] for word in i]
        seq_index = [word2index.get(word, word2index["<UNK>"]) for word in i]
        # 补全保持句子长度一致
        if len(seq_index) != seq_length:
            seq_index = seq_index + [0] * (seq_length-len(seq_index))
        inputs.append(seq_index)
    targets = [i for i in label]
    return inputs, targets


class MyLSTM(nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__()
        self.word_vec = nn.Embedding(vocab_size, embedding_size)
        # bidirectional双向LSTM
        self.bilstm = nn.LSTM(embedding_size, n_hidden, 1, bidirectional=True, batch_first=True)
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(n_hidden * 2))
        self.fc = nn.Linear(n_hidden * 2, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, input):
        embedding_input = self.word_vec(input)
        # 调换第一维和第二维度
        # embedding_input = embedding_input.permute(1, 0, 2)
        output, (h_n, c_n) = self.bilstm(embedding_input)

        # attention机制
        output = self.tanh(output)
        alpha = F.softmax(torch.matmul(output, self.w), dim=1).unsqueeze(-1)

        out = output * alpha
        out = torch.sum(out, 1)

        # # 使用正向LSTM与反向LSTM最后一个输出做拼接
        # encoding1 = torch.cat([h_n[0], h_n[1]], dim=1)  # dim=1代表横向拼接
        # # 使用双向LSTM的输出头尾拼接做文本分类
        # encoding2 = torch.cat([output[0], output[-1]], dim=1)
        fc_out = self.fc(out)
        fc_out = self.dropout(fc_out)

        return fc_out


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data_train, label_train, data_dev, label_dev, data_test, label_test = LoadData()
    # 分词
    train_cut, word2index, vocab = cut(data_train, make_vocab=True)
    dev_cut = cut(data_dev)
    test_cut = cut(data_test)
    # 参数
    vocab_size = len(word2index)
    # seq_size = len(seq)
    seq_length = max([len(i) for i in train_cut])
    print(seq_length)
    batch_size = 50
    embedding_size = 256
    num_classes = 2
    n_hidden = 512

    # 创建dataloader
    train_data, train_label = make_data(train_cut, label_train)
    train_data, train_label = Variable(torch.tensor(train_data, device=device)), Variable(
        torch.tensor(train_label, device=device))

    train_dataset = Data.TensorDataset(train_data, train_label)
    train_loader = Data.DataLoader(train_dataset, batch_size, shuffle=True)

    dev_data, dev_label = make_data(dev_cut, label_dev)
    dev_data, dev_label = Variable(torch.tensor(dev_data, device=device)), Variable(
        torch.tensor(dev_label, device=device))

    dev_dataset = Data.TensorDataset(dev_data, dev_label)
    dev_loader = Data.DataLoader(dev_dataset, batch_size, shuffle=True)

    test_data, test_label = make_data(test_cut, label_test)
    test_data, test_label = Variable(torch.tensor(test_data, device=device)), Variable(
        torch.tensor(test_label, device=device))

    test_dataset = Data.TensorDataset(test_data, test_label)
    test_loader = Data.DataLoader(test_dataset, batch_size, shuffle=True)

    if os.path.exists("./save/bilstm_att_drop.pth"):
        model = torch.load("./save/bilstm_att_drop.pth")
        model = model.to(device)
    else:
        model = MyLSTM()
        model = model.to(device=device)

        criterion = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=0.001)
        # model = model.to(device=device)
        # 开始训练
        for epoch in range(10):
            model.train()
            for train_batch_data, train_batch_label in train_loader:
                pre = model.forward(train_batch_data)
                train_loss = criterion(pre, train_batch_label)
                train_loss.backward()
                opt.step()
                opt.zero_grad()

            model.eval()
            all_pre = []
            all_tag = []
            with torch.no_grad():
                for dev_batch_data, dev_batch_label in dev_loader:
                    pre = model.forward(dev_batch_data)
                    dev_loss = criterion(pre, dev_batch_label)
                    pre = torch.argmax(pre, dim=-1).reshape(-1)
                    all_pre.extend(pre.detach().cpu().numpy().tolist())
                    all_tag.extend(dev_batch_label.detach().cpu().numpy().reshape(-1).tolist())
                # report = classification_report(all_tag, all_pre)
                # print(report)
            score = f1_score(all_tag, all_pre, average="micro")
            print(f"{epoch}:\tf1_score:{score:.3f}\ttrain_loss:{train_loss}\tdev_loss:{dev_loss}")
        # 保存模型
        torch.save(model, './save/bilstm_att_drop.pth')


    # 测试
    model.eval()
    all_pre = []
    all_pre_score_neg = []
    all_pre_score_pos = []
    all_tag = []
    with torch.no_grad():
        for test_batch_data, test_batch_label in test_loader:
            pre = model.forward(test_batch_data)
            # 负例预测分数
            pre_score_neg = [p[0] for p in pre.detach().cpu().numpy().tolist()]
            all_pre_score_neg.extend(pre_score_neg)
            # 正例预测分数
            pre_score_pos = [p[1] for p in pre.detach().cpu().numpy().tolist()]
            all_pre_score_pos.extend(pre_score_pos)
            pre = torch.argmax(pre, dim=-1).reshape(-1)
            all_pre.extend(pre.detach().cpu().numpy().tolist())
            all_tag.extend(test_batch_label.detach().cpu().numpy().reshape(-1).tolist())
        # report = classification_report(all_tag, all_pre)
        # print(report)
    p = precision_score(all_tag, all_pre)
    r = recall_score(all_tag, all_pre)
    f1 = f1_score(all_tag, all_pre)
    acc = accuracy_score(all_tag, all_pre)
    print(f"test_precision:{p:.10f}\ntest_recall:{r:.10f}\ntest_f1_score:{f1:.10f}\ntest_accuracy:{acc:.10f}")
    report = classification_report(all_tag, all_pre, digits=4)
    print(report)

    auc_p = roc_auc_score(all_tag, all_pre_score_pos)
    auc_n = roc_auc_score(all_tag, all_pre_score_neg)
    print(f"pos auc:{auc_p}\nneg auc:{auc_n}")


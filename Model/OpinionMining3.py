"""
@author:Yangt
@file:OpinionMining.py
@time:2022/05/01
@version:1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class OpinionMining3(nn.Module):
    def __init__(self, vocab_size, embedding_size, n_hidden, num_classes, dr_rate=0.5, bi=True):
        super(OpinionMining3, self).__init__()
        self.word_vec = nn.Embedding(vocab_size, embedding_size)
        self.bilstm = nn.LSTM(embedding_size, n_hidden, 1, bidirectional=bi, batch_first=True)
        self.bilstm_2 = nn.LSTM(embedding_size, n_hidden, 1, bidirectional=bi, batch_first=True)
        self.tanh = nn.Tanh()

        self.self_attn_target = nn.MultiheadAttention(n_hidden * 2, 1, batch_first=True)
        self.self_attn_words = nn.MultiheadAttention(n_hidden * 2, 1, batch_first=True)

        # self.pool = nn.MaxPool1d()
        # self.w = nn.Parameter(torch.zeros(n_hidden * 2))
        # self.w2 = nn.Parameter(torch.zeros(n_hidden * 2))
        # # 加性attention
        # self.Wq = nn.Linear(embedding_size, att_hn)
        # self.Wk = nn.Linear(n_hidden*2, att_hn)
        # self.Wv = nn.Linear(att_hn, 1)
        # 分类
        self.fc = nn.Linear(n_hidden * 2 * 2, num_classes)
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, input, target):
        # 词嵌入层
        embedding_input = self.word_vec(input)
        embedding_target = self.word_vec(target)
        # bilstm层
        output, (h_n, c_n) = self.bilstm(embedding_input)
        output = self.tanh(output)

        # output = self.dropout(output)

        # target的attention向量
        out_target, (h_n, c_n) = self.bilstm_2(embedding_target)
        out_target = self.tanh(out_target)

        # out_target = self.dropout(out_target)

        out_target, _ = self.self_attn_target(out_target, out_target, out_target)
        out_target = out_target.sum(1)
        # out_target = torch.max(out_target, dim=1).values
        # alpha = F.softmax(torch.matmul(out_target, self.w), dim=1).unsqueeze(-1)
        #
        # out_target = out_target * alpha
        # out_target = torch.sum(out_target, 1)

        # 计算target和隐状态之间的attention
        alpha_2 = F.softmax(torch.matmul(output, out_target.unsqueeze(-1)), dim=1)
        out = output * alpha_2
        out = torch.sum(out, 1)

        out_self, _ = self.self_attn_words(output, output, output)
        out_self = out_self.sum(1)
        # out_self = torch.mean(out_self, dim=1)
        # alpha_3 = F.softmax(torch.matmul(output, self.w2), dim=1).unsqueeze(-1)
        #
        # out_self = output * alpha_3
        # out_self = torch.sum(out_self, 1)

        # out_final = torch.add(out_self, out)

        out_final = torch.cat((out_self, out), dim=1)

        # print(out_final.size())

        # 分类
        fc_out = self.fc(out_final)
        fc_out = self.dropout(fc_out)

        return fc_out


if __name__ == '__main__':
    sent = [
        [4, 2, 6, 3, 0],
        [4, 1, 6, 5, 4],
        [4, 2, 6, 3, 0],
        [4, 2, 5, 3, 0],
    ]
    tr = [
        [4, 2],
        [1, 3],
        [7, 2],
        [6, 1],
    ]
    lb = [0, 1, 0, 0]

    sent = torch.tensor(sent, )
    tr = torch.tensor(tr)
    lb = torch.tensor(lb)

    model = OpinionMining3(12, 16, 7, 2, dr_rate=0.6)
    lab = model(sent, tr)
    for i in range(10000):
        criterion = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=0.01)
        lab = model(sent, tr)
        loss = criterion(lab, lb)
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(f"loss:{loss}")
    print(lab)
    # print(lb)



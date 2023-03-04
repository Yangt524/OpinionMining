"""
@author:Yangt
@file:attention_train.py
@time:2022/06/28
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

# from Model.OpinionMining2 import OpinionMining2
# from Model.OpinionMining3 import OpinionMining3
from Model.SelfAttention import SelfAttention
from DataPreprocessing import LoadData, cut, make_data, get_dataloader

torch.manual_seed(7)  # cpu
torch.cuda.manual_seed(7)  # gpu


def main(batch_size=50, embedding_size=300, n_hidden=300, num_classes=2, dr_rate=0.4, load=True,
         save=False, file_path='./save/self_atte.pth', data_path='./data/om/', epochs=40, learning_rate=5e-5):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    data_train, target_train, label_train, \
    data_dev, target_dev, label_dev, \
    data_test, target_test, label_test = LoadData(data_path)
    # 分词
    train_seq_cut, train_target_cut, word2index, vocab = cut(data_train, target_train, make_vocab=True)
    dev_seq_cut, dev_target_cut = cut(data_dev, target_dev)
    test_seq_cut, test_target_cut = cut(data_test, target_test)

    # 参数
    vocab_size = len(word2index)
    # seq_size = len(seq)
    MAX_SEQ = max([len(i) for i in train_seq_cut])
    MAX_TARGET = max([len(i) for i in train_target_cut])
    # print(seq_length)
    # 创建dataloader
    # train
    train_loader = get_dataloader(train_seq_cut, train_target_cut, label_train, word2index, MAX_SEQ, MAX_TARGET,
                                  device, batch_size)

    # dev
    dev_loader = get_dataloader(dev_seq_cut, dev_target_cut, label_dev, word2index, MAX_SEQ, MAX_TARGET, device,
                                batch_size)

    # test
    test_loader = get_dataloader(test_seq_cut, test_target_cut, label_test, word2index, MAX_SEQ, MAX_TARGET, device,
                                 batch_size)

    # 定义模型
    model = SelfAttention(vocab_size, embedding_size, num_classes, dr_rate=dr_rate)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=learning_rate)

    if os.path.exists(file_path) and load:
        state_dict = torch.load(file_path)
        model.load_state_dict(state_dict)
        model = model.to(device=device)
        # print(state_dict)
    else:
        model = model.to(device=device)
        # criterion = nn.CrossEntropyLoss()
        # opt = optim.Adam(model.parameters(), lr=0.001)
        # 开始训练
        for epoch in range(epochs):
            model.train()
            for train_batch_data, train_batch_target, train_batch_label in train_loader:
                pre = model(train_batch_data)
                train_loss = criterion(pre, train_batch_label)
                # opt.zero_grad()
                train_loss.backward()
                opt.step()
                opt.zero_grad()

            model.eval()
            all_pre = []
            all_tag = []
            with torch.no_grad():
                for dev_batch_data, dev_batch_target, dev_batch_label in dev_loader:
                    pre = model(dev_batch_data)
                    dev_loss = criterion(pre, dev_batch_label)
                    pre = torch.argmax(pre, dim=-1).reshape(-1)
                    all_pre.extend(pre.detach().cpu().numpy().tolist())
                    all_tag.extend(dev_batch_label.detach().cpu().numpy().reshape(-1).tolist())
                # report = classification_report(all_tag, all_pre)
                # print(report)
            score = f1_score(all_tag, all_pre, average="micro")
            print(f"{epoch}:\tf1_score:{score:.10f}\ttrain_loss:{train_loss}\tdev_loss:{dev_loss}")
        # 保存模型
        # torch.save(model, 'main.pth')
        # print(model.parameters())
        if save:
            torch.save(model.state_dict(), file_path)
        # print(model.state_dict())

    # 测试
    model.eval()
    all_pre = []
    all_pre_score_neg = []
    all_pre_score_pos = []
    all_tag = []
    with torch.no_grad():
        for test_batch_data, test_batch_target, test_batch_label in test_loader:
            pre = model(test_batch_data)
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
    f1 = f1_score(all_tag, all_pre, average='macro')
    acc = accuracy_score(all_tag, all_pre)
    print(f"test_precision:{p:.10f}\ntest_recall:{r:.10f}\ntest_f1_score:{f1:.10f}\ntest_accuracy:{acc:.10f}")
    report = classification_report(all_tag, all_pre, digits=4)
    print(report)
    # print(all_pre)
    # print(all_tag)
    return f1, acc


if __name__ == '__main__':
    main()

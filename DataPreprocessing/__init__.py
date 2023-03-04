import collections

import jieba
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as Data


def LoadData(data_path):
    """
    从文件读取数据
    :return:
    """
    train_df, dev_df, test_df = pd.read_csv(data_path + 'train.csv'), \
                                pd.read_csv(data_path + 'dev.csv'), \
                                pd.read_csv(data_path + 'test.csv')
    train, dev, test = np.array(train_df), np.array(dev_df), np.array(test_df)

    data_train = [train[i][2] for i in range(len(train))]
    label_train = [train[i][0] for i in range(len(train))]
    target_train = [train[i][1] for i in range(len(train))]

    data_dev = [dev[i][2] for i in range(len(dev))]
    label_dev = [dev[i][0] for i in range(len(dev))]
    target_dev = [dev[i][1] for i in range(len(dev))]

    data_test = [test[i][2] for i in range(len(test))]
    label_test = [test[i][0] for i in range(len(test))]
    target_test = [test[i][1] for i in range(len(test))]

    return data_train, target_train, label_train, data_dev, target_dev, label_dev, data_test, target_test, label_test


def cut(seq, target, make_vocab=False):
    seq_cut_list, tg_cut_list = [], []
    vocab_list = []
    for s, t in zip(seq, target):
        s_cut = list(jieba.cut(s))
        t_cut = list(jieba.cut(t))
        vocab_list = vocab_list + s_cut + t_cut
        seq_cut_list.append(s_cut)
        tg_cut_list.append(t_cut)

    if make_vocab:
        # 计数
        word2num = sorted(collections.Counter(vocab_list).items(), key=lambda item: item[1], reverse=True)
        # 去重
        vocab = [w[0] for w in word2num]
        word2index = {w[0]: i + 1 for i, w in enumerate(word2num)}
        word2index["<PAD>"] = 0
        word2index["<UNK>"] = len(word2index)
        return seq_cut_list, tg_cut_list, word2index, vocab
    else:
        return seq_cut_list, tg_cut_list


def make_data(seq, target, word2index, MAX_SEQ, MAX_TARGET):
    inputs, targets = [], []
    for s, t in zip(seq, target):
        s_index = [word2index.get(word, word2index["<UNK>"]) for word in s]
        t_index = [word2index.get(word, word2index["<UNK>"]) for word in t]
        if len(s_index) < MAX_SEQ:
            s_index = s_index + [0] * (MAX_SEQ - len(s_index))
        if len(t_index) < MAX_TARGET:
            t_index = t_index + [0] * (MAX_TARGET - len(t_index))
        inputs.append(s_index)
        targets.append(t_index)
    return inputs, targets


def get_dataloader(seq_cut, target_cut, label, word2index, MAX_SEQ, MAX_TARGET, device, batch_size):
    seq_index, target_index = make_data(seq_cut, target_cut, word2index, MAX_SEQ, MAX_TARGET)
    seq_index, target_index, label = Variable(torch.tensor(seq_index, device=device)), Variable(
        torch.tensor(target_index, device=device)), Variable(torch.tensor(label, device=device))
    dataset = Data.TensorDataset(seq_index, target_index, label)
    loader = Data.DataLoader(dataset, batch_size, shuffle=True)
    return loader


if __name__ == '__main__':
    pass

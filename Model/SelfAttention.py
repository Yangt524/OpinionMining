"""
@author:Yangt
@file:SelfAttention.py
@time:2022/06/28
@version:
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class SelfAttention(nn.Module):
    def __init__(self, vocab_size, embedding_size, num_classes, dr_rate=0.5):
        super(SelfAttention, self).__init__()
        self.word2vec = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.attention = nn.MultiheadAttention(embedding_size, 4, batch_first=True)

        self.fc = nn.Linear(embedding_size, num_classes)
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, input):
        embedding = self.word2vec(input)
        out, _ = self.attention(embedding, embedding, embedding)
        out = out.mean(1)

        fc_out = self.fc(out)
        fc_out = self.dropout(fc_out)

        return fc_out


if __name__ == '__main__':
    pass

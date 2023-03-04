"""
@author:Yangt
@file:TextCNN.py
@time:2022/06/07
@version:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, feature_size, max_text_len, window_sizes, num_class, dropout_rate):
        super(TextCNN, self).__init__()
        # self.is_training = True
        self.dropout_rate = dropout_rate
        # self.num_class = 2
        # self.use_element = config.use_element
        # self.config = config
        # vocab_size = 30147
        # embedding_size = 256
        # feature_size = 100
        # max_text_len = 35
        # window_sizes = [3, 4, 5, 6]

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_size,
                                      padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=embedding_size,
                                    out_channels=feature_size,
                                    kernel_size=h),
                          #                              nn.BatchNorm1d(num_features=config.feature_size),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=max_text_len-h+1))
            for h in window_sizes
        ])
        self.fc = nn.Linear(in_features=feature_size*len(window_sizes),
                            out_features=num_class)

    def forward(self, x):
        embed_x = self.embedding(x)
        # embed_x = x

        # print('embed size 1',embed_x.size())  # 32*35*256
# batch_size x text_len x embedding_size  -> batch_size x embedding_size x text_len
        embed_x = embed_x.permute(0, 2, 1)
        # print('embed size 2',embed_x.size())  # 32*256*35
        # out[i]:batch_size x feature_size*1
        out = [conv(embed_x) for conv in self.convs]
        # for o in out:
        #    print('o',o.size())  # 32*100*1
        out = torch.cat(out, dim=1)  # 对应第二个维度（行）拼接起来，比如说5*2*1,5*3*1的拼接变成5*5*1
        # print(out.size(1))  # 32*400*1
        out = out.view(-1, out.size(1))
        # print(out.size())  # 32*400
        # if not self.use_element:
        out = F.dropout(input=out, p=self.dropout_rate)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    pass

"""
@author:Yangt
@file:get_result2.py
@time:2022/06/21
@version:
"""
import sys
sys.path.append("..")
from ATTEBiLSTM2 import main
import itertools


if __name__ == '__main__':
    batch_size = [50]
    embedding_size = [256, 512]
    n_hidden = [256, 512]
    dr_rate = [0.3, 0.4, 0.5, 0.6]

    res = itertools.product(batch_size, embedding_size, n_hidden, dr_rate, repeat=1)
    # 结果转化为列表
    result = list(res)
    print(len(result))
    for bt, em, hn, dr in result:
        f1, acc = main(batch_size=bt, embedding_size=em, n_hidden=hn, dr_rate=dr, load=False, save=False,
                       data_path='../data/om/')
        with open('paras4.txt', 'a', encoding='utf8') as f:
            result = "--batch_size:" + str(bt) + "\t--embedding_size:" + str(em) + "\t--n_hidden:" + str(hn) + \
                     "\t--dr_rate:" + str(dr) + "\t--acc:" + str(acc) + "\t--f1:" + str(f1)
            f.write(result + '\n')



    # f1, acc = main(data_path='../')

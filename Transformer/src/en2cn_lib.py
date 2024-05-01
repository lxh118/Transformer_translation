# -*- coding: utf-8 -*-
"""
File: en2cn_lib
Author: little star
Created on 2024/1/30.
Project: Transformer
"""
import pickle
import random
import numpy as np
import torch
import os
from src.word2sequence import Word2Sequence

en_vocab_size = 25000
cn_vocab_size = 25000

lrd = True
earlyStopping = 4

epochs = 10
batch_size = 16
num_works = 4
train_ratio = 0.8
val_ratio = 0.1

data_path = r"D:\MyFile\DataSet\wmt2020\news-commentary-v15\news-commentary-v15.en-zh.tsv"


def init_seeds(seed=10):
    # 随机seed,保证实验可重复性
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 获取当前模块的路径
current_module_path = os.path.dirname(os.path.abspath(__file__))

# 相对路径
en_ws_path = "../model/en2cn/en_ws.pkl"
cn_ws_path = "../model/en2cn/cn_ws.pkl"
model_save_path = "../model/en2cn/model_net.pt"

# 构建绝对路径
en_ws_path = os.path.join(current_module_path, en_ws_path)
cn_ws_path = os.path.join(current_module_path, cn_ws_path)
model_save_path = os.path.join(current_module_path, model_save_path)

try:
    en_ws = pickle.load(open(en_ws_path, "rb"))
except FileNotFoundError:
    en_ws = Word2Sequence()
    pickle.dump(en_ws, open(en_ws_path, "wb"))

try:
    cn_ws = pickle.load(open(cn_ws_path, "rb"))
except FileNotFoundError:
    cn_ws = Word2Sequence()
    pickle.dump(cn_ws, open(cn_ws_path, "wb"))

INPUT_DIM = len(en_ws)  # 词典词数 + 4个特殊标记符
OUTPUT_DIM = len(cn_ws)  # 词典词数 + 4个特殊标记符

D_MODEL = 128
NUM_HEADS = 4
FF_HIDDEN_DIM = 512
NUM_LAYERS = 3
DROPOUT = 0.1
DECODER_MAX_LEN = 100  # Decoder输出句子的最大长度

if __name__ == "__main__":
    # print(cn_ws.count)
    print(cn_ws.dict)

# -*- coding: utf-8 -*-
"""
File: cn2en_lib
Author: little star
Created on 2024/2/1.
Project: Transformer
"""

import pickle
import os
from src.word2sequence import Word2Sequence
from src.en2cn_lib import init_seeds

init_seeds(seed=10)  # 服用en2cn_lib的函数

en_vocab_size = 100000
cn_vocab_size = 4500

lrd = True
earlyStopping = 4

epochs = 10
batch_size = 4
num_works = 4
train_size = 2000000
val_size = 400000
test_size = 400000
num_samples = train_size + val_size + test_size

en_data_path = r"D:\MyFile\DataSet\wmt2020\back-translation\news.en"

cn_data_path = r"D:\MyFile\DataSet\wmt2020\back-translation\news.translatedto.zh"

# 相对路径
en_ws_path = "../model/cn2en/en_ws.pkl"
cn_ws_path = "../model/cn2en/cn_ws.pkl"
model_save_path = "../model/cn2en/model_net.pt"

# 获取当前模块的路径
current_module_path = os.path.dirname(os.path.abspath(__file__))

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

INPUT_DIM = len(cn_ws)  # 词典词数 + 4个特殊标记符
OUTPUT_DIM = len(en_ws)  # 词典词数 + 4个特殊标记符

D_MODEL = 64
NUM_HEADS = 8
FF_HIDDEN_DIM = 256
NUM_LAYERS = 2
DROPOUT = 0.1
DECODER_MAX_LEN = 100  # Decoder输出句子的最大长度

if __name__ == "__main__":
    # print(cn_ws.count)
    print(cn_ws.dict)
    print(en_ws.dict)
    print(len(en_ws))

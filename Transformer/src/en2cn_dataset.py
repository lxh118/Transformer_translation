# -*- coding: utf-8 -*-
"""
File: en2zh_dataset
Author: little star
Created on 2024/1/31.
Project: Transformer
"""
import csv
import nltk
import jieba
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
from src import en2cn_lib


class En2CnDataset(Dataset):

    def __init__(self, dataPath=en2cn_lib.data_path):
        self.data_path = dataPath
        self.total_src = []
        self.total_trg = []

        # 读取包含多个 JSON 对象的文本文件
        with open(self.data_path, 'r', encoding="utf-8") as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t')

            # 读取每一行
            for row in tsvreader:
                # row 是一个包含英文和中文的列表
                src_data, trg_data = row
                # 跳过空白行
                if src_data == '' or trg_data == '':
                    continue
                self.total_src.append(src_data)
                self.total_trg.append(trg_data)

    def __getitem__(self, index):
        src_data = self.total_src[index]
        trg_data = self.total_trg[index]
        src_tokens = tokenizer(src_data, language="en")
        trg_tokens = tokenizer(trg_data, language="cn-word")

        src_tokens_length = len(src_tokens)
        trg_tokens_length = len(trg_tokens)

        return src_tokens, trg_tokens, src_tokens_length, trg_tokens_length

    def __len__(self):
        return len(self.total_src)


def get_dataloader(dataset, train_size, val_size, test_size, batch_size=en2cn_lib.batch_size,
                   num_workers=en2cn_lib.num_works, custom_collate_fn=None):
    """
    实现数据集的加载
    :param custom_collate_fn:用于处理序列数据的mini-batch
    :param dataset: (torch.utils.data.Dataset): 数据集对象，包含源语言和目标语言的数据对。
    :param train_size: (int): 训练集大小
    :param val_size: (int): 验证集大小
    :param test_size: (int): 测试集大小
    :param num_workers: 加载数据的进程数
    :param batch_size: 加载批次大小
    :return: 训练集、验证集、测试集、训练集数据用作制作词典
    """

    __train_dataset, __val_dataset, __test_dataset = random_split(dataset, [train_size, val_size, test_size])

    __train_dataloader = DataLoader(__train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                    collate_fn=custom_collate_fn, drop_last=True)
    __val_dataloader = DataLoader(__val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                  collate_fn=custom_collate_fn, drop_last=True)
    __test_dataloader = DataLoader(__test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                   collate_fn=custom_collate_fn, drop_last=True)

    return __train_dataloader, __val_dataloader, __test_dataloader


def tokenizer(content, language=None):
    """
    :param language: 需要分词的语言
    :param content: 句子
    :return: 清洗后的分词结果
    """
    if language == "en":
        tokens = [word.strip().lower() for word in nltk.word_tokenize(content)]
    elif language == "cn-character":
        tokens = [char for char in content]
    elif language == "cn-word":
        tokens = list(jieba.cut(content, cut_all=False))
    else:
        tokens = [word.strip().lower() for word in nltk.word_tokenize(content)]
        print("默认为english分词,如果为其他语言参数language需要输入,请检查该函数！！！")
    return tokens


def en2cn_collate_fn(batch):
    """
    自定义的collate_fn函数，用于处理序列数据的mini-batch

    :param batch: ([src_tokens,trg_tokens,src_tokens_length, trg_tokens_length],)
    :return:
    """

    # input 按长度降序排序
    batch = sorted(batch, key=lambda x: x[-1], reverse=True)
    src_tokens, trg_tokens, src_tokens_length, trg_tokens_length = zip(*batch)

    src_max_len = max(src_tokens_length)
    trg_max_len = max(trg_tokens_length)

    src_list = [en2cn_lib.en_ws.transform(i, max_len=src_max_len) for i in src_tokens]
    trg_list = [en2cn_lib.cn_ws.transform(i, max_len=trg_max_len) for i in trg_tokens]

    src_list = torch.LongTensor(src_list)
    trg_list = torch.LongTensor(trg_list)

    src_tokens_length = torch.LongTensor(src_tokens_length)
    trg_tokens_length = torch.LongTensor(trg_tokens_length)

    return src_list, trg_list, src_tokens_length, trg_tokens_length


if __name__ == "__main__":
    en2cn_dataset = En2CnDataset()

    en2cn_total_size = len(en2cn_dataset)
    en2cn_train_size = int(en2cn_lib.train_ratio * en2cn_total_size)
    en2cn_val_size = int(en2cn_lib.val_ratio * en2cn_total_size)
    en2cn_test_size = en2cn_total_size - en2cn_train_size - en2cn_val_size

    en2cn_train_dataloader, en2cn_val_dataloader, en2cn_test_dataloader = get_dataloader(en2cn_dataset,
                                                                                         en2cn_train_size,
                                                                                         en2cn_val_size,
                                                                                         en2cn_test_size,
                                                                                         custom_collate_fn=en2cn_collate_fn)
    print(len(en2cn_train_dataloader))
    for idx, (srcTokens, trgTokens, srcTokens_length, trgTokens_length) in enumerate(en2cn_train_dataloader):
        if idx % 10000 == 0:
            print(f"Sample {idx + 1} \n Src Tokens: {srcTokens} \n Trg Tokens: {trgTokens} \n "
                  f"Src Length:{srcTokens_length}\n TrgLength: {trgTokens_length}")

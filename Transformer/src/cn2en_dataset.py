# -*- coding: utf-8 -*-
"""
File: cn2en_dataset
Author: little star
Created on 2024/2/1.
Project: Transformer
"""
import random

import torch
from torch.utils.data import Dataset
from src import cn2en_lib
from src.en2cn_dataset import tokenizer, get_dataloader


class Cn2EnDataset(Dataset):

    def __init__(self, cn_dataPath=cn2en_lib.cn_data_path, en_dataPath=cn2en_lib.en_data_path
                 , num_samples=cn2en_lib.num_samples):
        self.cn_data_path = cn_dataPath
        self.en_data_path = en_dataPath
        self.total_src = []
        self.total_trg = []

        self.num_samples = num_samples

        # 读取包含多个 JSON 对象的文本文件并随机选择
        with open(self.cn_data_path, 'r', encoding="utf-8") as cn_file, open(self.en_data_path, 'r', encoding="utf-8") as en_file:
            lines = list(zip(cn_file, en_file))
            sampled_lines = random.sample(lines, min(num_samples, len(lines)))

            for cn_line, en_line in sampled_lines:
                self.total_src.append(cn_line.strip())
                self.total_trg.append(en_line.strip())

        # print(len(self.total_src), len(self.total_trg))

    def __getitem__(self, index):
        src_data = self.total_src[index]
        trg_data = self.total_trg[index]
        src_tokens = tokenizer(src_data, language="cn-word")
        trg_tokens = tokenizer(trg_data, language="en")

        src_tokens_length = len(src_tokens)
        trg_tokens_length = len(trg_tokens)

        return src_tokens, trg_tokens, src_tokens_length, trg_tokens_length

    def __len__(self):
        return len(self.total_src)


# noinspection DuplicatedCode
def cn2en_collate_fn(batch):
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

    src_list = [cn2en_lib.cn_ws.transform(i, max_len=src_max_len) for i in src_tokens]
    trg_list = [cn2en_lib.en_ws.transform(i, max_len=trg_max_len) for i in trg_tokens]

    src_list = torch.LongTensor(src_list)
    trg_list = torch.LongTensor(trg_list)

    src_tokens_length = torch.LongTensor(src_tokens_length)
    trg_tokens_length = torch.LongTensor(trg_tokens_length)

    return src_list, trg_list, src_tokens_length, trg_tokens_length


if __name__ == '__main__':
    cn2en_dataset = Cn2EnDataset()

    cn2en_train_size = cn2en_lib.train_size
    cn2en_val_size = cn2en_lib.val_size
    cn2en_test_size = cn2en_lib.test_size

    cn2en_train_dataloader, cn2en_val_dataloader, cn2en_test_dataloader = get_dataloader(cn2en_dataset,
                                                                                         cn2en_train_size,
                                                                                         cn2en_val_size,
                                                                                         cn2en_test_size,
                                                                                         batch_size=cn2en_lib.batch_size,
                                                                                         num_workers=cn2en_lib.num_works,
                                                                                         custom_collate_fn=cn2en_collate_fn)
    print(len(cn2en_train_dataloader))
    for idx, (srcTokens, trgTokens, srcTokens_length, trgTokens_length) in enumerate(cn2en_train_dataloader):
        if idx % 10000 == 0:
            print(f"Sample {idx + 1} \n Src Tokens: {srcTokens} \n Trg Tokens: {trgTokens} \n "
                  f"Src Length:{srcTokens_length}\n TrgLength: {trgTokens_length}")

    pass

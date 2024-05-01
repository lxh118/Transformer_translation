# -*- coding: utf-8 -*-
"""
File: build_vocab
Author: little star
Created on 2024/1/31.
Project: Transformer
"""
import pickle
from tqdm import tqdm

from src import cn2en_lib, en2cn_lib
from src.cn2en_dataset import Cn2EnDataset
from src.en2cn_dataset import En2CnDataset
from src.word2sequence import Word2Sequence

cn2en_lib.init_seeds(seed=10)


def en2cn_builb_vocabe():
    en_ws = Word2Sequence()
    cn_ws = Word2Sequence()
    en2cn_dataset = En2CnDataset()
    for idx in tqdm(range(len(en2cn_dataset))):
        srcTokens, trgTokens, srcTokens_length, trgTokens_length = en2cn_dataset[idx]

        en_ws.fit(srcTokens)
        cn_ws.fit(trgTokens)
    en_ws.build_vocab(minCount=5, max_features=en2cn_lib.en_vocab_size)
    cn_ws.build_vocab(minCount=5, max_features=en2cn_lib.cn_vocab_size)
    pickle.dump(en_ws, open(en2cn_lib.en_ws_path, "wb"))
    pickle.dump(cn_ws, open(en2cn_lib.cn_ws_path, "wb"))

    return len(en_ws), len(cn_ws)


def cn2en_builb_vocabe():
    en_ws = Word2Sequence()
    cn_ws = Word2Sequence()
    cn2en_dataset = Cn2EnDataset()
    for idx in tqdm(range(len(cn2en_dataset))):
        srcTokens, trgTokens, srcTokens_length, trgTokens_length = cn2en_dataset[idx]

        cn_ws.fit(srcTokens)
        en_ws.fit(trgTokens)

    cn_ws.build_vocab(minCount=5, max_features=cn2en_lib.cn_vocab_size)
    en_ws.build_vocab(minCount=5, max_features=cn2en_lib.en_vocab_size)

    pickle.dump(en_ws, open(cn2en_lib.en_ws_path, "wb"))
    pickle.dump(cn_ws, open(cn2en_lib.cn_ws_path, "wb"))

    return len(en_ws), len(cn_ws)


if __name__ == "__main__":
    en_len, cn_len = en2cn_builb_vocabe()
    print(en_len, cn_len)
    # en_len, cn_len = cn2en_builb_vocabe()
    # print(en_len, cn_len)
    pass

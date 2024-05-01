# -*- coding: utf-8 -*-
"""
File: word2sequence
Author: little star
Created on 2024/1/31.
Project: Transformer
"""


class Word2Sequence:
    """
    实现的是：构建词典，实现方法把句子转化为数字序列和其翻转

    使用示例：

    ws = Word2Sequence()
    ws.fit(["我", "是", "谁"])
    ws.fit(["我", "是", "你"])

    ws.build_vocab(minCount=0, maxCount=None, max_features=None)
    print(ws.count)
    print(ws.dict)

    ret = ws.transform(["我", "爱", "中国", "我"], max_len=20)  # 词典转换数字表示
    print(ret)
    ret = ws.inverse_transform(ret)  # 数字转换成对应单词
    print(ret)
    """

    def __init__(self):

        self.UNK_TAG = "UNK"
        self.PAD_TAG = "PAD"
        self.BOS_TAG = "BOS"
        self.EOS_TAG = "EOS"

        self.UNK = 0
        self.PAD = 1
        self.SOS = 2
        self.EOS = 3

        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD,
            self.BOS_TAG: self.SOS,
            self.EOS_TAG: self.EOS,
        }
        self.count = {}  # 统计词频
        self.inverse_dict = {}

    def fit(self, sentence):
        """
        把单个句子保存到dict中
        :param sentence: [word1,word2...]
        :return:
        """
        for word in sentence:
            self.count[word] = self.count.get(word, 0) + 1

    def build_vocab(self, minCount=5, maxCount=None, max_features=None):
        """
        生成词典
        :param minCount: 最小出现的次数
        :param maxCount: 最大的次数
        :param max_features:一共保留多少个词语
        :return:
        """
        # 删除count中词频小于min的word
        if minCount is not None:
            self.count = {word: value for word, value in self.count.items() if value > minCount}
        # 删除count中词频大于max的word
        if maxCount is not None:
            self.count = {word: value for word, value in self.count.items() if value < maxCount}
        # 限制保留的词语数
        if max_features is not None:
            temp = sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_features]
            self.count = dict(temp)

        for word in self.count:
            self.dict[word] = len(self.dict)  # word为key,value为字典长度,从4开始

    def transform(self, sentence, max_len=100):
        """
        把句子转化为序列
        :param sentence: [word1,word2...]
        :param max_len: int,当前批次句子的最大长度
        :return:
        """
        paddings = [self.PAD_TAG] * (max_len - len(sentence))

        sentence = [self.BOS_TAG] + sentence + [self.EOS_TAG]

        sentence = sentence + paddings

        sentence = [self.dict.get(word, self.UNK) for word in sentence]

        return sentence

    def inverse_transform(self, indices):
        """
        把序列转化为句子
        :param indices:[1，2，3，4]
        :return:
        """
        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))
        return [self.inverse_dict.get(idx) for idx in indices]

    def __len__(self):
        return len(self.dict)


if __name__ == '__main__':
    ws = Word2Sequence()
    ws.fit(["我", "是", "谁"])
    ws.fit(["我", "是", "你"])

    ws.build_vocab(minCount=0, maxCount=None, max_features=None)
    print(ws.count)
    print(ws.dict)

    ret = ws.transform(["我", "爱", "中国", "我"], max_len=20)  # 词典转换数字表示
    print(ret)
    ret = ws.inverse_transform(ret)  # 数字转换成对应单词
    print(ret)
    pass

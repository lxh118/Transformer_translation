from transformer_class import Seq2SeqTransformer

import torch
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, size, padding_idx, unk_idx, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.unk_idx = unk_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        assert x.size(-1) == self.size  # Check the size of the last dimension
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2 + 1e-6))  # Ignore padding and UNK
        true_dist.scatter_(-1, target.data.unsqueeze(-1), self.confidence)  # Use -1 as the dimension
        # print(true_dist)
        true_dist[:, :, self.padding_idx] = 0  # Ignore PAD
        true_dist[:, :, self.unk_idx] = 0  # Ignore UNK
        # print(true_dist)
        # print(x)
        # 计算非零元素的数量
        non_pad_unk_positions = (target.data != self.padding_idx) & (target.data != self.unk_idx)
        num_non_pad_unk_elements = non_pad_unk_positions.sum()

        # 计算平均 KL 散度
        return self.criterion(x, true_dist) / num_non_pad_unk_elements.float()


# 例子
target_vocab_size = 4  # 你的目标词汇表大小
padding_index = 1  # padding 的索引
unk_index = 0  # UNK 的索引

logits = torch.tensor([[[0.8, 0.0, 0.2, 0.0], [0.7, 0.3, 0.0, 0.0]], [[0.8, 0.2, 0.0, 0.0], [0.3, 0.7, 0.0, 0.0]]])
targets = torch.tensor([[0, 2], [1, 2]])

criterion = LabelSmoothingLoss(size=target_vocab_size, padding_idx=padding_index, unk_idx=unk_index, smoothing=0.1)

loss = criterion(logits, targets)
print(loss)

if __name__ == '__main__':
    # Example for a translation task
    input_dim = 10000  # Vocabulary size for both source and target languages
    d_model = 512
    num_heads = 8
    ff_hidden_dim = 2048
    output_dim = 10000  # Vocabulary size for the target language
    num_layers = 6
    dropout = 0.1

    model = Seq2SeqTransformer(input_dim, output_dim, d_model, num_heads, ff_hidden_dim, num_layers, dropout)
    print(model)

    pass

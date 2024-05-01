# -*- coding: utf-8 -*-
"""
File: my_transformer
Author: little star
Created on 2024/1/30.
Project: Transformer
"""
import importlib
import numpy as np
import torch
from torch import nn
from torch.functional import F
from nltk.translate.bleu_score import corpus_bleu
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from src import transformer_utils
from src import draw_image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 训练gpu


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
        return self.criterion(x, true_dist)


class Seq2SeqTransformer(nn.Module):
    """
    Seq2SeqTransformer model combining TransformerEncoder and TransformerDecoder.
    """

    def __init__(self, input_dim, output_dim, d_model, num_heads, ff_hidden_dim, num_layers, dropout=0.1, **kwargs):
        """
        Initialize the Seq2SeqTransformer model.

        :param input_dim: The input dimension of the model.
        :param output_dim: The output dimension of the model.
        :param d_model: The model's feature dimension.
        :param num_heads: The number of attention heads in each Transformer block.
        :param ff_hidden_dim: The hidden dimension of the Feedforward layer in each Transformer block.
        :param num_layers: The number of Transformer blocks in the model.
        :param dropout: Dropout rate.
        """
        super(Seq2SeqTransformer, self).__init__()
        self.is_training = kwargs.get('is_training', True)
        self.lib = importlib.import_module(f"src.{kwargs.get('selected_lib', 'en2cn')}_lib")
        self.num_heads = num_heads
        self.encoder = transformer_utils.TransformerEncoder(input_dim, d_model, num_heads, ff_hidden_dim, d_model,
                                                            num_layers, dropout)
        self.decoder = transformer_utils.TransformerDecoder(output_dim, d_model, num_heads, ff_hidden_dim,
                                                            num_layers, dropout, is_training=self.is_training)

    def forward(self, src, trg, custom_mask):
        """
        Forward pass of the Seq2SeqTransformer model.

        :param src: Input tensor for the encoder.
        :param trg: Input tensor for the decoder.
        :param custom_mask: Custom mask tensor for attention weights in the encoder.
        :return: Output tensor after the Seq2SeqTransformer model.
        """
        src_padding_mask = src.unsqueeze(1).unsqueeze(1)
        enc_output = self.encoder(src, src_padding_mask, custom_mask)
        if self.is_training:
            trg_padding_mask = trg.unsqueeze(1).unsqueeze(1)
            dec_output = self.decoder(trg, enc_output, trg_padding_mask, src_padding_mask, custom_mask)
            prob = F.log_softmax(dec_output, dim=-1)  # [batcj_size,trg_len,output_dim]
            return prob
        else:
            # greddy decoder
            # best_sequence = self.greddy_decoder(trg, enc_output, src_padding_mask, custom_mask)

            # beam decoder
            best_sequence = self.beam_search_decoder(trg, enc_output, src_padding_mask, custom_mask, beam_width=5)
            return best_sequence

    def greddy_decoder(self, trg, enc_output, src_padding_mask, custom_mask):
        for i in range(self.lib.DECODER_MAX_LEN):
            trg_padding_mask = trg.unsqueeze(1).unsqueeze(1)
            # print(trg_padding_mask.shape)
            dec_output = self.decoder(trg, enc_output, trg_padding_mask, src_padding_mask, custom_mask)
            prob = F.log_softmax(dec_output, dim=-1)  # [batcj_size,trg_len,output_dim]

            _, output_sentences = torch.max(prob, dim=-1, keepdim=False)
            # print(output_sentences)
            next_word = output_sentences.data[:, -1].unsqueeze(-1)
            # print(next_word)
            trg = torch.cat([trg, next_word], dim=1)
        return trg[:, 1:]

    def beam_search_decoder(self, trg, enc_output, src_padding_mask, custom_mask, beam_width=5):

        score = torch.zeros(trg.size(0), dtype=torch.float).to(device)
        beam_outputs = [(trg, score)]  # List to store current beam sequences and their scores

        for i in range(self.lib.DECODER_MAX_LEN):
            all_candidates = []

            for trg_seq, score in beam_outputs:
                trg_padding_mask = trg_seq.unsqueeze(1).unsqueeze(1)

                dec_output = self.decoder(trg_seq, enc_output, trg_padding_mask, src_padding_mask, custom_mask)
                prob = F.log_softmax(dec_output, dim=-1)
                topk_probs, topk_indices = torch.topk(prob[:, -1, :], k=beam_width, dim=-1)

                for j in range(beam_width):
                    next_word = topk_indices[:, j].unsqueeze(-1)
                    new_seq = torch.cat([trg_seq, next_word], dim=1)
                    new_score = score - topk_probs[:, j]  # Negative log likelihood as score
                    all_candidates.append((new_seq, new_score))
            # print(all_candidates)
            # Select top beam_width sequences
            ordered = sorted(all_candidates, key=lambda x: x[1].mean())
            beam_outputs = ordered[:beam_width]

        # Return the best sequence
        best_seq, _ = min(beam_outputs, key=lambda x: x[1].mean())
        return best_seq[:, 1:]  # Remove the start token

    @staticmethod
    def cal_bleu(output_sentences, trgTokens):
        output_sentences = output_sentences.tolist()
        trgTokens = trgTokens.tolist()

        translation = []  # 翻译结果
        for sentence_i in range(len(output_sentences)):
            current_sentence = []

            for word_i in range(0, len(output_sentences[sentence_i])):
                now_word = output_sentences[sentence_i][word_i]
                if now_word == 3:  # EOS:3 结束符
                    break
                current_sentence.append(now_word)
            translation.append(current_sentence)

        trg_sentences = []  # 目标翻译
        for sentence_i in range(len(trgTokens)):
            current_sentence = []
            for word_i in range(1, len(trgTokens[sentence_i])):
                now_word = trgTokens[sentence_i][word_i]
                if now_word == 3:  # EOS:3 结束符
                    break
                current_sentence.append(now_word)
            trg_sentences.append([current_sentence])

        # print(len(trg_sentences), len(translation))

        train_bleu_score = corpus_bleu(trg_sentences, translation,
                                       weights=(0.8, 0.2))
        return train_bleu_score  # Remove the start token

    @staticmethod
    def trainer(my_model, epochs, train_dataloader, val_dataloader, my_optimizer, lib):
        train_losses = []
        val_losses = []
        train_bleu_scores = []
        val_bleu_scores = []

        # 学习率衰减
        schedule = ReduceLROnPlateau(my_optimizer, 'min', factor=0.3, patience=1, min_lr=1e-10,
                                     verbose=True)
        min_val_loss = np.inf
        delay = 0  # 早停计数器

        custom_mask = None  # Assuming a simple masking for the encoder

        criterion = LabelSmoothingLoss(size=lib.OUTPUT_DIM, padding_idx=1, unk_idx=0, smoothing=0.1)

        for epoch in range(epochs):
            my_model.train()
            my_model.is_training = True
            train_loss = 0.0
            train_bleu_score = 0.0
            for idx, (srcTokens, trgTokens, srcTokens_length, trgTokens_length) in tqdm(enumerate(train_dataloader),
                                                                                        total=len(train_dataloader),
                                                                                        desc="训练"):
                srcTokens = srcTokens.to(device)
                trgTokens = trgTokens.to(device)

                my_optimizer.zero_grad()

                model_output = my_model(srcTokens, trgTokens[:, :-1], custom_mask)  # [log(P(index))]
                # print(model_output.shape, trgTokens.shape)

                # 做（目标值*概率对数）求和取负操作,在根据batchsize做平均,忽略padding
                # loss = F.nll_loss(model_output.contiguous().view(-1, model_output.shape[-1]),trgTokens[:, 1:].reshape(-1), ignore_index=1)

                # 使用KL散度计算损失
                loss = criterion(model_output, trgTokens[:, 1:])

                loss.backward()
                torch.nn.utils.clip_grad_norm_(my_model.parameters(), max_norm=1)  # 梯度裁剪
                my_optimizer.step()

                train_loss += loss.item()
                _, output_sentences = torch.max(model_output, dim=-1, keepdim=False)
                # print(output_sentences.shape, trgTokens.shape)
                if idx % 1000 == 0:
                    print(output_sentences, trgTokens)
                    print(lib.cn_ws.inverse_transform(output_sentences[0].tolist()))
                    print(lib.cn_ws.inverse_transform(trgTokens[0].tolist()))

                # 计算 BLEU 分数
                train_bleu_score += my_model.cal_bleu(output_sentences, trgTokens[:, 1:])
                # print(train_bleu_score)

            train_loss = train_loss / len(train_dataloader)
            train_bleu_score = train_bleu_score / len(train_dataloader)

            print(f"Epoch {epoch}: train loss {train_loss:10.6f}, train_bleu_score {train_bleu_score:7.4f}")

            train_losses.append(train_loss)
            train_bleu_scores.append(train_bleu_score)

            my_model.eval()  # Set the model to evaluation mode
            val_loss = 0
            val_bleu_score = 0.0
            with torch.no_grad():
                for idx, (srcTokens, trgTokens, srcTokens_length, trgTokens_length) in tqdm(enumerate(val_dataloader),
                                                                                            total=len(val_dataloader),
                                                                                            desc="验证"):
                    srcTokens = srcTokens.to(device)
                    trgTokens = trgTokens.to(device)  # 实际的目标值

                    model_output = my_model(srcTokens, trgTokens[:, :-1], custom_mask)  # [log(P(index))]

                    # 做（目标值*概率对数）求和取负操作,在根据batchsize做平均
                    # loss = F.nll_loss(model_output.contiguous().view(-1, model_output.shape[-1]),
                    #                   trgTokens[:, 1:].reshape(-1), ignore_index=1)

                    loss = criterion(model_output, trgTokens[:, 1:])
                    val_loss += loss.item()
                    _, output_sentences = torch.max(model_output, dim=-1, keepdim=False)

                    # 计算 BLEU 分数
                    val_bleu_score += my_model.cal_bleu(output_sentences, trgTokens[:, 1:])

            val_loss = val_loss / len(val_dataloader)
            val_bleu_score = val_bleu_score / len(val_dataloader)

            print(f"val loss {val_loss:10.6f}, val_bleu_score {val_bleu_score:7.4f}")

            val_losses.append(val_loss)
            val_bleu_scores.append(val_bleu_score)

            if lib.lrd:
                schedule.step(val_loss)

            if val_loss < min_val_loss:
                torch.save(my_model.state_dict(), lib.model_save_path)
                min_val_loss = val_loss
                print(f"Update min_val_loss to {min_val_loss:10.6f}")
                delay = 0
            else:
                delay = delay + 1

            if delay > lib.earlyStopping:
                break

        # 绘制loss折线图
        draw_image.plot_loss(train_losses, val_losses)
        draw_image.plot_acc(train_bleu_scores, val_bleu_scores)

    @staticmethod
    def tester(my_model, test_dataloader, lib):
        my_model.load_state_dict(torch.load(lib.model_save_path))

        custom_mask = None  # Assuming a simple masking for the encoder

        my_model.eval()  # Set the model to evaluation mode
        my_model.is_training = False

        test_bleu_score = 0.0
        with torch.no_grad():
            for idx, (srcTokens, trgTokens, srcTokens_length, trgTokens_length) in tqdm(enumerate(test_dataloader),
                                                                                        total=len(test_dataloader),
                                                                                        desc="测试"):
                input_trg = torch.full((lib.batch_size, 1), 2).to(device)  # 0:UNK 1:PAD 2:SOS 3:EOS
                srcTokens = srcTokens.to(device)
                trgTokens = trgTokens.to(device)  # 实际的目标值

                output_sentences = my_model(srcTokens, input_trg, custom_mask)  # 已经是输出的序列句子了

                if idx % 100 == 0:
                    print(output_sentences, trgTokens)
                    print(lib.cn_ws.inverse_transform(output_sentences[0].tolist()))
                    print(lib.cn_ws.inverse_transform(trgTokens[0, 1:].tolist()))

                # 计算 BLEU 分数
                test_bleu_score += my_model.cal_bleu(output_sentences, trgTokens[:, 1:])

        test_bleu_score = test_bleu_score / len(test_dataloader)
        return test_bleu_score


if __name__ == '__main__':
    pass

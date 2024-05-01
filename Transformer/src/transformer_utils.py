# -*- coding: utf-8 -*-
"""
File: transformer_utils
Author: little star
Created on 2024/1/28.
Project: Transformer
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module to add positional information to input embeddings.
    """

    def __init__(self, d_model, max_len=512):
        """
        Initialize the PositionalEncoding module.

        :param d_model: The model's feature dimension.
        :param max_len: The maximum length of input sequences.
        """
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Forward pass of the PositionalEncoding module.

        :param x: Input tensor.
        :return: Output tensor with positional encoding added.
        """
        x = x + self.pe[:x.size(0), :]
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Module for the TransformerEncoder model.
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize the MultiHeadAttention module.

        :param d_model: The input feature dimension.
        :param num_heads: The number of attention heads.
        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Linear transformations for query, key, and value
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # Final linear layer for output after attention
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask, padding_mask):
        """
        Forward pass of the MultiHeadAttention module.

        :param padding_mask:
        :param query: Query tensor.
        :param key: Key tensor.
        :param value: Value tensor.
        :param mask: Optional mask tensor for attention weights.
        :return: Output tensor after attention.
        """
        batch_size = query.shape[0]

        # Ensure the data type consistency with the linear layer's weight
        query = self.query(query).to(self.fc_out.weight.dtype)
        key = self.key(key).to(self.fc_out.weight.dtype)
        value = self.value(value).to(self.fc_out.weight.dtype)

        # Split into heads
        Q = query.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = key.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = value.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # print(Q.shape, K.shape, V.shape)
        # Attention scores
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim).float())
        # print(energy.shape)
        # print(mask, padding_mask)
        if mask is not None:
            energy = energy.masked_fill(mask == 1, float('-inf'))
        if padding_mask is not None:
            energy = energy.masked_fill(padding_mask == 1, float('-inf'))
        # Attention weights
        attention = torch.nn.functional.softmax(energy, dim=-1)

        # Output
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.d_model)
        x = self.fc_out(x)

        return x


class TransformerEncoderBlock(nn.Module):
    """
    TransformerEncoder Block module containing Multi-Head Attention and Feedforward layers.
    """

    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout=0.1):
        """
        Initialize the TransformerEncoderBlock module.

        :param d_model: The input feature dimension.
        :param num_heads: The number of attention heads.
        :param ff_hidden_dim: The hidden dimension of the Feedforward layer.
        :param dropout: Dropout rate.
        """
        super(TransformerEncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_padding_mask, custom_mask):
        """
        Forward pass of the TransformerEncoderBlock module.

        :param x: Input tensor.
        :param src_padding_mask: Padding mask tensor for self-attention.
        :param custom_mask: Custom mask tensor for self-attention.
        :return: Output tensor after TransformerEncoderBlock.
        """
        # Multi-head attention
        attention_output = self.attention(x, x, x, custom_mask, src_padding_mask)
        # Add and normalize
        x = x + self.dropout(attention_output)
        x = self.norm1(x)

        # Feedforward network
        ffn_output = self.ffn(x)
        # Add and normalize
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    """
    TransformerEncoder model composed of multiple TransformerBlocks.
    """

    def __init__(self, input_dim, d_model, num_heads, ff_hidden_dim, output_dim, num_layers, dropout=0.1):
        """
        Initialize the TransformerEncoder model.

        :param input_dim: The input dimension of the model.
        :param d_model: The model's feature dimension.
        :param num_heads: The number of attention heads in each TransformerEncoderBlock.
        :param ff_hidden_dim: The hidden dimension of the Feedforward layer in each TransformerEncoderBlock.
        :param output_dim: The output dimension of the model.
        :param num_layers: The number of TransformerBlocks in the model.
        :param dropout: Dropout rate.
        """
        super(TransformerEncoder, self).__init__()
        self.num_heads = num_heads
        self.embedding = nn.Embedding(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_blocks = nn.ModuleList(
            [TransformerEncoderBlock(d_model, num_heads, ff_hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_padding_mask, custom_mask):
        """
        Forward pass of the TransformerEncoder model.

        :param src_padding_mask: Src padding mask tensor for attention weights.
        :param src: Input tensor.
        :param custom_mask: Optional mask tensor for attention weights.
        :return: Output tensor after the TransformerEncoder model.
        """

        # Embedding layer
        x = self.embedding(src)
        # Add positional encoding
        x = self.positional_encoding(x)
        # TransformerEncoder blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, src_padding_mask, custom_mask)
        # Global average pooling
        # x = self.fc_out(x.mean(dim=1))
        return x


class TransformerDecoderBlock(nn.Module):
    """
    TransformerDecoder Block module containing Multi-Head Attention and Feedforward layers.
    """

    def __init__(self, d_model, num_heads, ff_hidden_dim, dropout=0.1):
        """
        Initialize the TransformerDecoderBlock module.

        :param d_model: The input feature dimension.
        :param num_heads: The number of attention heads.
        :param ff_hidden_dim: The hidden dimension of the Feedforward layer.
        :param dropout: Dropout rate.
        """
        super(TransformerDecoderBlock, self).__init__()
        self.num_heads = num_heads
        self.masked_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, trg_padding_mask, src_padding_mask, trg_mask, custome_mask):
        """
        Forward pass of the TransformerDecoderBlock module.

        :param trg_mask: Trg mask tensor for self-attention.
        :param custome_mask:Custome mask tensor for self-attention.
        :param src_padding_mask:Padding mask tensor for src self-attention.
        :param x: Input tensor.
        :param enc_output: Output tensor from the encoder.
        :param trg_padding_mask: Padding mask tensor for self-attention.
        :return: Output tensor after TransformerDecoderBlock.
        """

        # Masked self-attention
        masked_attention_output = self.masked_attention(x, x, x, trg_mask, trg_padding_mask)
        x = x + self.dropout(masked_attention_output)
        x = self.norm1(x)

        # Encoder-decoder attention
        # print(x.shape, enc_output.shape)
        attention_output = self.attention(x, enc_output, enc_output, custome_mask, src_padding_mask)
        x = x + self.dropout(attention_output)
        x = self.norm2(x)

        # Feedforward network
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm3(x)

        return x


class TransformerDecoder(nn.Module):
    """
    TransformerDecoder model composed of multiple TransformerDecoderBlocks.
    """

    def __init__(self, output_dim, d_model, num_heads, ff_hidden_dim, num_layers, dropout=0.1, **kwargs):
        """
        Initialize the TransformerDecoder model.

        :param output_dim: The output dimension of the model.
        :param d_model: The model's feature dimension.
        :param num_heads: The number of attention heads in each TransformerDecoderBlock.
        :param ff_hidden_dim: The hidden dimension of the Feedforward layer in each TransformerDecoderBlock.
        :param num_layers: The number of TransformerDecoderBlocks in the model.
        :param dropout: Dropout rate.
        """
        super(TransformerDecoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.transformer_blocks = nn.ModuleList(
            [TransformerDecoderBlock(d_model, num_heads, ff_hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.embedding = nn.Embedding(output_dim, d_model)  # Add embedding layer
        self.positional_encoding = PositionalEncoding(d_model)  # Add positional encoding layer
        self.fc_out = nn.Linear(d_model, output_dim)
        self.is_training = kwargs.get('is_training', False)

    def forward(self, trg, enc_output, trg_padding_mask, src_padding_mask, custome_mask):
        """
        Forward pass of the TransformerDecoder output_dimmodel.
        :param trg_padding_mask: Trg padding mask tensor for attention weights.
        :param custome_mask:Custome mask tensor for attention weights.
        :param src_padding_mask:Src padding mask tensor for attention weights.
        :param trg: Input tensor for the decoder (target sequence).
        :param enc_output: Output tensor from the encoder.
        :return: Output tensor after the TransformerDecoder model.
        """
        # future mask
        if self.is_training:
            batch_size, num_steps = trg.shape
            trg_mask = torch.triu(torch.ones((num_steps, num_steps)), diagonal=1)
            trg_mask = trg_mask.unsqueeze(0).unsqueeze(0).to(trg.device)
        else:
            trg_mask = None

        # Embedding layer
        x = self.embedding(trg)
        # Add positional encoding
        x = self.positional_encoding(x)
        # TransformerDecoder blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, enc_output, trg_padding_mask, src_padding_mask, trg_mask, custome_mask)
        output = self.fc_out(x)
        # print(x.shape)
        return output

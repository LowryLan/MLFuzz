# -*- codeing = utf-8 -*-
# @Time : 2023/5/5 8:28
# @Author : Lowry
# @File : tsfm
# @Software : PyCharm

import torch
import torch.nn as nn
import data
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(SelfAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # 初始化查询、键、值映射矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 初始化线性变换矩阵
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, d_model = x.size()

        # 计算查询、键、值向量
        q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 计算得分矩阵
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))

        # 计算注意力权重
        attn_weights = nn.functional.softmax(scores, dim=-1)

        # 计算加权和向量
        attn_output = torch.matmul(attn_weights, v)

        # 将多头注意力输出拼接起来
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # 线性变换投影到输出向量空间
        output = self.W_o(attn_output)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        # 初始化多头自注意力模块
        self.self_attention = SelfAttention(d_model, n_heads)

        # 初始化线性变换矩阵
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        # batch_size, seq_len, d_model = x.size()

        # 计算多头自注意力输出
        attn_output = self.self_attention(x)

        # 线性变换投影到输出向量空间
        # output = self.W_o(attn_output)

        # return output
        return attn_output


def test(path=None):
    """
    Get weight sequence of byte sequence

    :parameter path: seed path
    :return: weight list of byte sequence
    """
    # path = './programs/libxml/in/'
    multihead_attn = MultiHeadAttention(d_model=10000, n_heads=8)
    x_arr = data.get_byte(path)
    x_arr = np.array([x_arr])
    x_arr = x_arr.astype(np.float32)
    x_ten = torch.tensor(x_arr)
    output = multihead_attn(x_ten)
    output = output[0].detach().numpy()
    print(output)
    return output


test('./programs/libxml/in/')

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


def generate_weight(path=None):
    """
    Get weight metric of byte metric

    :parameter path: seed path
    :return: weight list of byte sequence
    """
    # path = './programs/libxml/in/'
    if path is None:
        return 0

    multihead_attn = MultiHeadAttention(d_model=10000, n_heads=8)

    byte_arr, file_list, file_len = data.get_byte(path)
    byte_arr = np.array([byte_arr])
    byte_arr = byte_arr.astype(np.float32)
    byte_ten = torch.tensor(byte_arr)

    output = multihead_attn(byte_ten)
    output = output[0].detach().numpy()

    if write_to_file(output, file_list, file_len):
        return 1
    else:
        return -1


def write_to_file(w_matrix=None, file_list=None, file_len=None):
    """
    Write weight metric info to file

    :param file_len: length of byte sequence in one seed
    :param file_list: file name list
    :parameter w_matrix: [np.array] weight matrix of seed byte
    :return: 1 or -1
    """

    if w_matrix is None or file_list is None:
        return -1

    with open('weight_info', 'w') as f:
        for i in range(len(file_list)):
            # print(file_list[i] + ": " + str(w_matrix[i][0]))
            j = file_len[i]
            weight_info = ['1' if w_matrix[i][l] > 0 else '-1' for l in range(j)]
            f.write(','.join(weight_info) + '|' + file_list[i] + '\n')
    # print(a)
    # print(b)
    return 1


# print(generate_weight('./programs/libxml/in/'))

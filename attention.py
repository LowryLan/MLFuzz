# -*- codeing = utf-8 -*-
# @Time : 2023/5/5 8:28
# @Author : Lowry
# @File : tsfm
# @Software : PyCharm

import torch
import torch.nn as nn
import data
import numpy as np
import goodSeed
import similarity


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


def split_list(byte_arr=None, file_list=None, file_len=None):
    """
    draw seeds with 10000+ bytes
    :parameter byte_arr: byte sequence of seed
    :parameter file_list: seed name list
    :parameter file_len: length of byte sequence

    :return: byte_arr_new, file_list_new, byte_arr_full, file_list_full
    """

    byte_arr_new = []  # byte list with 10000- byte
    file_list_new = []  # seed name with 10000- byte
    byte_arr_full = []  # byte list with 10000+ byte
    file_list_full = []  # seed name with 10000+ byte
    file_len_new = []  # length of byte sequence with 10000- byte
    file_len_full = []  # length of byte sequence with 10000+ byte

    for i in range(len(file_len)):
        if file_len[i] <= 10000:
            byte_arr_new.append(byte_arr[i])
            file_list_new.append(file_list[i])
            file_len_new.append(file_len[i])
        else:
            byte_arr_full.append(byte_arr[i])
            file_list_full.append(file_list[i])
            file_len_full.append(file_len[i])
    byte_arr_new = np.array(byte_arr_new)
    byte_arr_full = np.array(byte_arr_full)
    return byte_arr_new, file_list_new, byte_arr_full, file_list_full, file_len_new, file_len_full


def generate_weight(path=None, project=None):
    """
    Get weight metric of byte metric

    :param project: project directory name
    :parameter path: seed path
    :return: weight list of byte sequence
    """
    if path is None or project is None:
        return 0

    d_model = 10000
    flag = 0  # 1: the longest length of seed is more than 10000 || 0: shorter than 10000
    byte_arr, file_list, file_len = data.get_byte(path)

    byte_arr0 = byte_arr

    if max(file_len) > 10000:
        byte_arr_new, file_list_new, byte_arr_full, file_list_full, file_len_new, file_len_full = \
            split_list(byte_arr=byte_arr, file_list=file_list, file_len=file_len)
        flag = 1
        byte_arr0 = byte_arr_new
        byte_arr = np.array([byte_arr_new])
    else:
        file_list_full = []
        byte_arr = np.array([byte_arr])
    byte_arr = byte_arr.astype(np.float32)
    byte_ten = torch.tensor(byte_arr)

    multihead_attn = MultiHeadAttention(d_model=d_model, n_heads=8)

    output = multihead_attn(byte_ten)
    output = output[0].detach().numpy().tolist()  # get weight info

    # only use seeds with 10000- bytes
    if flag == 1:
        print('flag is 1')
        file_list = file_list_new
        file_len = file_len_new

    similarity_list = similarity.similarity_re(byte_array=byte_arr0, seed_list=file_list, max_len=max(file_len))

    # 按照文件id重新排序
    index_list = []
    file_len_new = file_len
    file_list_new = file_list
    output_new = output

    # for i in range(len(file_list)):
    #     index = int(file_list[i].split(',')[0].split(':')[1])
    #     index_list.append(index)
    # for i in range(file_list_len_orig):
    #     if i in index_list:
    #         file_list_new.append(file_list[index_list.index(i)])
    #         file_len_new.append(file_len[index_list.index(i)])
    #         output_new.append(output[index_list.index(i)])
    # 排序 end

    dir_path = './programs/' + project + '/out/queue'
    # good_seed = goodSeed.main_prt(flag0=flag0, dir_path=dir_path, cur_path=cur_path)
    # print(good_seed)

    # if len(file_len_new) <= 1000:
    #     seed_num = 700
    # elif len(file_len_new) <= 2000:
    #     seed_num = 1000
    # elif len(file_len_new) > 2000:
    #     seed_num = 1500
    seed_num = int(len(file_len_new) * 0.5)

    # if write_to_file(output_new, file_list_new, file_len_new, project, file_list_full, good_seed):
    if write_to_file(output_new, file_list_new, file_len_new, project, file_list_full, similarity_list, seed_num):
        return 1
    else:
        return -1


def write_to_file(w_matrix=None, file_list=None, file_len=None, project=None, file_list_full=None, similarity_list=None, seed_num=None):
    """
    Write weight metric info to file

    :param seed_num: select seed number
    :param good_seed: only chose these seed
    :param file_list_full:
    :param project: project directory name
    :param file_len: length of byte sequence in one seed
    :param file_list: file name list
    :parameter w_matrix: [np.array] weight matrix of seed byte
    :return: 1 or -1
    """

    if w_matrix is None or file_list is None:
        return -1

    sign = str(int(seed_num * 0.7))

    with open('./afl-lowry/weight_info', 'w') as f:
    # with open('./programs/' + project + '/weight_info', 'w') as f:
        for t in range(seed_num):
            temp = similarity_list[t]
            i = file_list.index(temp)
            j = file_len[i]
            # file_name = './programs/' + project + '/out/queue/' + file_list[i]
            weight_info = ['1' if w_matrix[i][l] > 0 else '-1' for l in range(j)]
            f.write(','.join(weight_info) + '|/home/lowry/Documents/myFuzz/MLFuzz/programs/' + project + '/out/queue/' +
                    file_list[i] + '|' + sign + '\n')
            # f.write(','.join(weight_info) + '|' + str(file_len[i]) + '|' + file_list[i] + '\n')
    return 1


def write_to_file_neuzz(w_matrix=None, file_list=None, file_len=None, project=None, file_list_full=None, sign=None):
    """
    Write weight metric info to file for neuzz.c

    :param good_seed: only chose these seed
    :param file_list_full:
    :param project: project directory name
    :param file_len: length of byte sequence in one seed
    :param file_list: file name list
    :parameter w_matrix: [np.array] weight matrix of seed byte
    :return: 1 or -1
    """

    if w_matrix is None or file_list is None:
        return -1

    with open('./programsForNeuzz/' + project + '/gradient_info_p', 'w') as f:
        for i in range(len(file_list)):
            j = file_len[i]
            if file_list[i] in file_list_full or (file_list[i][2] == ':' and sign == 1):
                continue
            else:
                if sign == 0:
                    weight_forward = ['1' if w_matrix[i][l] > 0 else '-1' for l in range(j)]
                else:
                    weight_forward = ['-1' if w_matrix[i][l] > 0 else '1' for l in range(j)]
                weight_info = [str(int(w_matrix[i][l])) if w_matrix[i][l] > 0 else str(-1 * int(w_matrix[i][l])) for l in
                               range(j)]
            f.write(','.join(weight_info) + '|' + ','.join(weight_forward) + '|' + './seeds/' + file_list[i] + '\n')
            # f.write(','.join(weight_info) + '|' + str(file_len[i]) + '|' + file_list[i] + '\n')
    return 1


# print(generate_weight('./programs/readelf/in/', project='readelf'))

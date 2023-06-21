# -*- codeing = utf-8 -*-
# @Time : 2023/6/19 10:54
# @Author : Lowry
# @File : goodSeed
# @Software : PyCharm

import os
import mooModel

seed_path = './programs/readelf/out/queue'
cur_seed = 18         # 新旧种子分叉点（当前种子数量可用作下一轮种子选择的分叉点）

chosen_num = []
offspring_num = []
flag = 0


# 读取种子文件名
def get_seed_name(dir_path=None):
    """
    读取种子文件

    """
    X_file_name = []  # 用于存储被选出的种子文件名
    files = os.listdir(dir_path)
    for file in files:  # 遍历文件夹
        if file[-6:] != '.state':
            X_file_name.append(dir_path + '/' + file)  # 存储所有种子文件名
    index_list = []
    seed_list = []
    for i in range(len(X_file_name)):
        index = int(X_file_name[i].split(',')[0].split(':')[1])
        index_list.append(index)
    for i in range(len(index_list)):
        seed_list.append(X_file_name[index_list.index(i)])
    return seed_list


# 分离新旧种子
def draw_new_seed(seed_list=None):
    if cur_seed == len(seed_list):
        return seed_list, seed_list
    else:
        return seed_list[:cur_seed], seed_list[cur_seed:]


# 解析种子文件名，并更新offspring_num
def parse_seed_name(seed_list=None):
    """
    Parse seed name to update offspring_number
    :parameters seed_list: new seed list

    :return : void
    """
    global offspring_num
    offspring_num = []
    src_list = []
    for i in range(len(seed_list)):
        offspring_num.append(0)
        seed_name = seed_list[i]
        seed_info = seed_name.split(',')
        seed_id = int(seed_info[0].split(':')[1])
        if seed_id == 0:
            continue
        else:
            src_id_str = seed_info[1].split(':')[1]
            if '+' in src_id_str:
                src_id_1 = int(src_id_str.split('+')[0])
                src_id_2 = int(src_id_str.split('+')[1])
                src_list.append(src_id_1)
                src_list.append(src_id_2)
            else:
                src_id = int(seed_info[1].split(':')[1])
                src_list.append(src_id)
    for j in src_list:
        offspring_num[j] += 1


# 构建MOO模型,选择旧种子中的最优种子
def get_moo_seed(old_seeds=None):
    global offspring_num, chosen_num, cur_seed
    task1 = chosen_num[:cur_seed]
    task2 = offspring_num[:cur_seed]
    new_seed_num = len(offspring_num) - cur_seed
    good_seeds = []
    global flag
    num = 1
    if new_seed_num < 1000:
        if flag == 0:
            if len(old_seeds) != 1:
                num = 2
        else:
            if 1000 - new_seed_num < len(old_seeds):
                num = 1000 - new_seed_num
            else:
                return old_seeds
        mp = mooModel.MyPareto(task1=task1, task2=task2, num=num)
        good_seeds_index = mp.get_seeds()
        for i in good_seeds_index:
            good_seeds.append(old_seeds[i])
    return good_seeds


# 更新chosen_seeds
def update_chosen_num(final_list=None, seed_list=None):
    global chosen_num
    for i in range(len(seed_list)):
        if seed_list[i] in final_list:
            chosen_num[i] += 1


# main function
def main_prt(flag0=None, dir_path=None, cur_path=None):
    global flag, cur_seed, chosen_num
    flag = flag0
    if cur_path:
        cur_seed = cur_path
        for i in range(cur_seed):
            chosen_num.append(1)

    """ 读取种子名 """
    seed_list = get_seed_name(dir_path=dir_path)

    """ 更新chosen_num（补长） """
    for i in range(cur_seed, len(seed_list)):
        chosen_num.append(0)

    """ 分离新旧种子 """
    old_seeds, new_seeds = draw_new_seed(seed_list=seed_list)

    """ 解析种子文件名，并更新offspring_num """
    parse_seed_name(seed_list=seed_list)

    """ 构建MOO模型,选择旧种子中的最优种子 """
    good_seeds = get_moo_seed(old_seeds=old_seeds)

    """ 和新种子拼接并排序 """
    final_list = good_seeds + new_seeds

    """ 更新chosen_num（加1） """
    update_chosen_num(final_list=final_list, seed_list=seed_list)

    """ 更新cur_seed """
    cur_seed = len(seed_list)

    return final_list


# print(main_prt(flag0=0, dir_path=seed_path, cur_path=19))

# -*- codeing = utf-8 -*-
# @Time : 2023/6/20 8:36
# @Author : Lowry
# @File : moo-model
# @Software : PyCharm

class MyPareto:
    """
    construct MOO Model
    """
    def __init__(self, task1, task2, num):
        """
        initial MOO model
        :param task1: 横轴, 存储种子被选择次数，越少越好，-1。高优先级
        :param task2: 纵轴，存储种子生成的有趣种子数，越多越好，+1
        :param num:   需要返回的种子数量
        """
        self.task1 = task1
        self.task2 = task2
        self.num = num
        self.index = []             # 用于存储task2从大到小排序的索引
        self.task1_matrix = []      # 将task1作为横坐标构建的初步坐标系信息（即，无纵坐标信息）

    def fill_index(self):
        """
        填充 index
        """
        task2_sorted = sorted(self.task2, reverse=True)                          # 从大到小排序
        i = 0
        while i < len(task2_sorted):
            temp = task2_sorted[i]
            index_temp = [t for t, x in enumerate(self.task2) if x == temp]        # 依次查找task2_sorted元素在task2中的索引
            for j in index_temp:
                self.index.append(j)
            i += len(index_temp)

    def get_x_position(self, value):
        """
        获取value在坐标系中的横坐标
        :param value: task2中的值
        :returns 横坐标值
        """
        x_pos = -1
        for i in range(len(self.task1_matrix)):
            if value in self.task1_matrix[i]:
                x_pos = i
        return x_pos

    def sort_task1_matrix(self):
        """
        将task1_matrix中的每行根据task2中的信息从大到小排序
        """
        for i in range(len(self.task1_matrix)):
            list_temp = self.task1_matrix[i]                            # 获取task1_matrix当前行
            lista = [self.task2[j] for j in list_temp]            # 获取task2中list_temp所含元素位置的值
            lista_sorted = sorted(lista, reverse=True)                  # 获取lista从大到小排序后的结果
            # list_re =
            list_re = [lista.index(j) for j in lista_sorted]            # 将原队列进行排序，从前到后在task2映射的值为从大到小
            self.task1_matrix[i] = list_re                              # 将排序结果替换原列表

    def fill_task1_matrix(self):
        """
        构建类坐标系，用于存储每列的种子索引。不会有重复！
        """
        task1_matrix_col = max(self.task1)
        for i in range(1, task1_matrix_col + 1):
            index_temp = [j for j, x in enumerate(self.task1) if x == i]
            self.task1_matrix.append(index_temp)
        self.sort_task1_matrix()                                        # 对每列中的元素进行排序

    def get_seeds(self):
        """
        主函数，返回所选种子位于task1中的索引
        :return index_re: 最终结果
        """
        index_re = []
        list_pos = []                               # 存储每列当前读取的位置
        for i in range(len(self.task1_matrix)):     # 初始化list_pos（0填充）
            list_pos.append(0)

        max_t1 = max(self.task1)
        k = 0                                       # 记录已选种子数量，< num

        self.fill_index()
        for i in range(len(self.index)):
            if self.index[i] not in index_re:
                index_re.append(self.index[i])
                k += 1
                j = i
                if j == len(self.index) - 1 or k == self.num:
                    break
                else:
                    j += 1
                temp = self.task1[self.index[i]]                             # 记录index[j]中所指task1的值，必须比他大才能被选入队列
                while max_t1 != temp and k < self.num and j < len(self.index) - 1:
                    if self.index[j] not in index_re and self.task1[self.index[j]] > temp:
                        index_re.append(self.index[j])
                        k += 1
                        temp = self.task1[self.index[j]]
                    j += 1
            if k == self.num:
                break

        return index_re


# task1 = [2, 2, 3, 1, 1, 1, 2, 3, 1, 2]
# task2 = [4, 3, 4, 0, 1, 0, 6, 5, 9, 6]
# mp = MyPareto(task1=task1, task2=task2, num=5)
# print(mp.get_seeds())


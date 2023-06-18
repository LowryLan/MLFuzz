# -*- codeing = utf-8 -*-
# @Time : 2023/5/7 16:34
# @Author : Lowry
# @File : recoder
# @Software : PyCharm

import os
AFL_SHOWMAP_BINARY_PATH = './afl-showmap'


def record_edge(testcase_dir, binary_file_path, base_cmd, save_path):
    open(save_path, "w")
    edge = set()
    pp = 0
    for root, dirs, files in os.walk(testcase_dir):
        if root[-5:] == 'queue':
            for file in files:
                # print(pp)
                pp = pp + 1
                testcase_path = os.path.join(root, file)
                cmd = f"{AFL_SHOWMAP_BINARY_PATH} -q -e -o /dev/stdout -m 512 -t 500 {binary_file_path} {base_cmd} {testcase_path}"
                out = os.popen(cmd).readlines()
                for o in out:
                    # print(o)
                    # edge_item = o.strip()
                    edge_item = o.split(':')[0]
                    edge.add(edge_item)
    with open(save_path, "a") as save_file:
        save_file.write(f"edge: {len(edge)}\n")
        # for e in edge:
        #     save_file.write(f"{e}\n")


# record_edge('/home/lowry/Documents/myFuzz/MLFuzz/programs/zlib/out/',
#             '/home/lowry/Documents/afl-program/zlib/miniunz',
#             '-o',
#             './edge_cov_zlib')

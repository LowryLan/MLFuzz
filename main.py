# -*- coding = utf-8 -*-
# @Time : 2023/5/12 13:46
# @Author : Lowry
# @File : main
# @Software : PyCharm

import sys
import socket
import attention

HOST = '127.0.0.1'
PORT = 12012


def get_args():
    """
    get arguments from terminal
    example: ./main.py libxml[argv1] xmllint[argv2]

    :return: project & PUT
    """
    project = sys.argv[1]       # directory name of project
    PUT = sys.argv[2]           # file name of program under test
    cur_path = sys.argv[3]      # fuzzing position
    return project, PUT, int(cur_path)


def main():
    print('begin py mode')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(1)
    conn, addr = sock.accept()
    project, PUT, cur_path = get_args()
    print(f"{project} is the program under test")
    print(f"cur_path is: {cur_path}")
    path = './programs/' + project + '/out/queue/'
    print('connected by MLFuzz execution module ' + str(addr))
    attention.generate_weight(path=path, project=project, flag0=0, cur_path=cur_path)
    conn.sendall(b"start")
    print("send success")
    while True:
        data = conn.recv(1024)
        if not data:
            break
        else:
            print(f'connected, sign is {data}')
            attention.generate_weight(path=path, project=project, flag0=1, cur_path=None)
            print('generate complete')
            conn.sendall(b"yesss")            # send to MLFuzz
    conn.close()


if __name__ == '__main__':
    main()

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
    return project, PUT


def main():
    print('begin py mode')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(1)
    conn, addr = sock.accept()
    project, PUT = get_args()
    path = './programs/' + project + '/out/queue/'
    print('connected by neuzz execution moduel ' + str(addr))
    attention.generate_weight(path, project)
    conn.sendall(b"start")
    print("send success")
    while True:
        data = conn.recv(1024)
        if not data:
            break
        else:
            print('connected')
            attention.generate_weight(path)
            print('generate complete')
            conn.sendall(b"yesss")
    conn.close()


if __name__ == '__main__':
    main()

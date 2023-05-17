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


def main():
    print('begin py mode')
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, PORT))
    sock.listen(1)
    conn, addr = sock.accept()
    print('connected by neuzz execution moduel ' + str(addr))
    attention.generate_weight('./programs/readelf/out/queue/')
    conn.sendall(b"start")
    print("send success")
    while True:
        data = conn.recv(1024)
        if not data:
            break
        else:
            print('connected')
            attention.generate_weight('./programs/readelf/out/queue/')
            print('generate complete')
            conn.sendall(b"yesss")
    conn.close()


if __name__ == '__main__':
    main()

# -*- coding:utf-8 -*-
import socket
serverAddr = "127.0.0.1"
serverPort = 12345

client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_sock.connect((serverAddr, serverPort))

while True:
    clientInput= input(' please input > ')
    client_sock.send(clientInput.encode('utf-8'))
    response = client_sock.recv(1024)
    print(response.decode('utf-8'))

client_sock.close()
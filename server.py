import socket
import threading
import random

host = '26.64.220.173'
port = 12345
clients = {}
room_codes = {}
client_addresses = {}  # dictionary để lưu địa chỉ ip của mỗi cl
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((host, port))
server.listen()
print(f"ok {host}:{port}")

def handle_client(conn, addr):
    print(f"{addr}")
    client_ip = addr[0]
    while True:
        try:
            room_code = conn.recv(1024).decode('utf-8')
            if room_code == 'NEW':
                room_code = str(random.randint(1000, 9999))
                while room_code in room_codes:
                    room_code = str(random.randint(1000, 9999))
                room_codes[room_code] = []
            if room_code not in room_codes:
                conn.send("wrong code".encode('utf-8'))
                continue
            conn.send(f"Connected to room {room_code}".encode('utf-8'))
            break
        except:
            conn.close()
            return
    
    room_codes[room_code].append(conn)
    clients[conn] = room_code
    client_addresses[conn] = client_ip  
    
    if len(room_codes[room_code]) == 2:
        
        ip_1, ip_2 = (client_addresses[conn] for conn in room_codes[room_code])
        room_codes[room_code][0].send(f"OTHER_USER_IP:{ip_2}".encode('utf-8'))
        room_codes[room_code][1].send(f"OTHER_USER_IP:{ip_1}".encode('utf-8'))

    while True:
        try:
            msg = conn.recv(1024)
            if msg:
                broadcast(msg, room_code, conn)
        except:
            conn.close()
            room_codes[room_code].remove(conn)
            del clients[conn]
            del client_addresses[conn]
            break

def broadcast(msg, room_code, sender):
    for client in room_codes[room_code]:
        if client != sender:
            try:
                client.send(msg)
            except:
                client.close()
                room_codes[room_code].remove(client)
                del clients[client]
                del client_addresses[client]

def start():
    while True:
        conn, addr = server.accept()
        thread = threading.Thread(target=handle_client, args=(conn, addr))
        thread.start()
        print(f"cur connect {threading.active_count() - 1}")

start()

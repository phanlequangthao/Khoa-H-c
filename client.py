import socket
import threading
import subprocess
import time
host = '26.213.15.26'
port = 12345
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((host, port))

def receive_messages():
    while True:
        try:
            message = client.recv(1024).decode('utf-8')
            if "OTHER_USER_IP:" in message:
                other_user_ip = message.split(":")[1]
                subprocess.Popen(["python", "mainlstm.py"])
                time.sleep(7)
                subprocess.Popen(["python", "client_camera.py"])
                print("done")
            else:
                print(message)
        except Exception as e:
            print("An error occurred!", e)
            client.close()
            break

def send_messages():
    while True:
        msg = input('->')
        client.send(msg.encode('utf-8'))


room_code = input("Enter room code or type 'NEW' for a new room: ")
client.send(room_code.encode('utf-8'))
server_message = client.recv(1024).decode('utf-8')
print(server_message)
if "Connected to room" in server_message:
    threading.Thread(target=receive_messages).start()
    send_messages()
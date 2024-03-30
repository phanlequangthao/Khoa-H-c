import socket
import threading
import subprocess

# Server IP and port
host = '26.64.220.173'
port = 12345

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((host, port))

def receive_messages():
    while True:
        try:
            message = client.recv(1024).decode('utf-8')
            if "OTHER_USER_IP:" in message:
                # Extract the other user's IP from the message
                other_user_ip = message.split(":")[1]
                # Use the extracted IP to run the necessary subprocesses
                subprocess.Popen(["python", "client_camera.py", other_user_ip])
                subprocess.Popen(["python", "svaudio.py"])
                subprocess.Popen(["python", "claudio.py", "--host_ip", other_user_ip])
                subprocess.Popen(["python", "mainlstm.py"])
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
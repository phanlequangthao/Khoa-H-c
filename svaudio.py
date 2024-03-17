import socket, cv2, pickle,struct,time
import pyshine as ps
import subprocess
import re
def get_ipv4_address(adapter_name):
    # Chạy lệnh ipconfig và lấy đầu ra
    result = subprocess.run(["ipconfig"], stdout=subprocess.PIPE, text=True)
    output = result.stdout

    # Tạo biểu thức chính quy để tìm kiếm IPv4
    pattern = rf"{adapter_name}.*?IPv4 Address. . . . . . . . . . . : (\d+\.\d+\.\d+\.\d+)"
    
    # Tìm kiếm địa chỉ IPv4 trong kết quả
    match = re.search(pattern, output, re.DOTALL)
    if match:
        return match.group(1)  # Trả về địa chỉ IPv4 tìm được
    else:
        return "IPv4 address not found."

adapter_name = "Ethernet adapter Radmin VPN"
ipv4_address = get_ipv4_address(adapter_name)


mode =  'send'
name = 'SERVER TRANSMITTING AUDIO'
audio,context= ps.audioCapture(mode=mode)
ps.showPlot(context,name)
server_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
host_ip = ipv4_address
port = 4982
backlog = 5
socket_address = (host_ip,port)
print('r',socket_address,'...')
server_socket.bind(socket_address)
server_socket.listen(backlog)
while True:
	client_socket,addr = server_socket.accept()
	print('ip:',addr)
	if client_socket:
		while(True):
			frame = audio.get()
			a = pickle.dumps(frame)
			message = struct.pack("Q",len(a))+a
			client_socket.sendall(message)
	else:
		break
client_socket.close()
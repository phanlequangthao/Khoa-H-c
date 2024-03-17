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

# Sử dụng
adapter_name = "Ethernet adapter Radmin VPN"
ipv4_address = get_ipv4_address(adapter_name)
print(f"{ipv4_address}")

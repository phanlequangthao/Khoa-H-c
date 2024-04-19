import imagiz
import cv2

# if len(sys.argv) != 2:
#     print("Usage: python client_camera.py <server_ip>")
#     sys.exit(1)

# Use the provided IP address as the server IP
server_ip = "26.157.245.17"

client = imagiz.Client("cc1", server_ip=server_ip)
# vid = cv2.VideoCapture(0)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
while True:
    try:
        frame = cv2.imread('shared_frame.jpg', 1)
        r, image = cv2.imencode('.jpg', frame, encode_param)
        client.send(image)
    except cv2.error as e:
        print(f"Error: {e}")
        continue
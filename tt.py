import cv2

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Central Program', frame)
    cv2.imwrite('shared_frame.jpg', frame) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# room_code = input("Enter room code or type 'NEW' for a new room: ")
#     client.send(room_code.encode('utf-8'))
#     server_message = client.recv(1024).decode('utf-8')
#     print(server_message)
#     if "Connected to room" in server_message:
#         app = QApplication(sys.argv)
#         window = Ham_Chinh()
#         window.setWindowTitle('MainApp')
#         window.show()
#         sys.exit(app.exec_())
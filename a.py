from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel
from PyQt5.QtCore import QThread, pyqtSignal
import socket
import sys
from PyQt5.QtWidgets import QLineEdit
class MessageReceiver(QThread):
    update_text = pyqtSignal(str)  # Tín hiệu để cập nhật label text2 trong GUI

    def __init__(self, socket):
        super().__init__()
        self.socket = socket

    def run(self):
        while True:
            try:
                message = self.socket.recv(1024).decode('utf-8')
                self.update_text.emit(message)  # Gửi tín hiệu cập nhật GUI
            except Exception as e:
                print(f"Error receiving message: {e}")
                break

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("MainLSTM App")
        self.setGeometry(100, 100, 400, 200)

        # Tạo label để hiển thị tin nhắn nhận được
        self.label_text2 = QLabel("Received Messages Here", self)
        self.label_text2.resize(300, 50)
        self.label_text2.move(50, 100)

        # Tạo label để nhập tin nhắn gửi đi
        self.label_text1 = QLabel("Type your message here", self)
        self.label_text1.resize(300, 50)
        self.label_text1.move(50, 30)

        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText("Type your message here")
        self.text_input.move(50, 30)
        self.text_input.resize(300, 20)    
        
        # Tạo nút gửi tin nhắn
        self.btn_send = QPushButton('Send', self)
        self.btn_send.move(150, 150)
        self.btn_send.clicked.connect(self.sendMessage)

        # Thiết lập kết nối và bắt đầu thread nhận tin nhắn
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect(('26.64.220.173', 12345))  # Địa chỉ và port của server
        self.receiver_thread = MessageReceiver(self.client_socket)
        self.receiver_thread.update_text.connect(self.updateText2)
        self.receiver_thread.start()

    def sendMessage(self):
        message = self.text_input.text()
        if message: 
            formatted_message = f"MSG:{message}"
            print(f"Sending message: {formatted_message}")
            self.client_socket.send(formatted_message.encode('utf-8'))
        else:
            print("No message to send.")



    def updateText2(self, message):
        print(f"Received message: {message}")  # In ra tin nhắn nhận được để debug
        self.label_text2.setText(message)


def main():
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.uic import loadUi
import cv2
import sys
import imagiz
import speech_recognition as sr
from moviepy.editor import  ImageSequenceClip
import mediapipe as mp
import pandas as pd
import numpy as np
import pickle
from win32com.client import Dispatch
import os
import socket
import threading
import subprocess
<<<<<<< HEAD
import tensorflow as tf
import base64
import time
from keras.models import load_model
mphands = mp.solutions.hands
hands = mphands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils
=======
mp_drawing = mp.solutions.drawing_utils 
mp_holistic = mp.solutions.holistic
>>>>>>> d2db99074efd303f87130d642d8077403ccdf2c1
speak = Dispatch("SAPI.SpVoice").Speak
class SpeechToVideoThread(QThread):
    video = pyqtSignal(QImage)
    audioTextChanged = pyqtSignal(str)
    def __init__(self, img_dir, video_output_path):
        super(SpeechToVideoThread, self).__init__()
        self.img_dir = img_dir
<<<<<<< HEAD
        self.video_output_path = video_output_path
        self.audio_text = ""
        self.is_recording = False
    def run(self):
        r = sr.Recognizer()
        m = sr.Microphone()
        print("A moment of silence, please...")
        with m as source: r.adjust_for_ambient_noise(source)
        print("Set minimum energy threshold to {}".format(r.energy_threshold))
        while self.is_recording:
            print("Say something!")
            with m as source: audio = r.listen(source)
            print("Got it! Now to recognize it...")
            try:
                self.audio_text = r.recognize_google(audio)
                print("You said {}".format(self.audio_text))
                self.create_video_from_text()
                self.audioTextChanged.emit(self.audio_text)
            except sr.UnknownValueError:
                print("Oops! Didn't catch that")
            except sr.RequestError as e:
                print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
=======
        self.img_dir2 = r"D:\img2"
        self.video_output_path = video_output_path
        self.video_output_path2 = r"output_video2.mp4"
        self.audio_text = ""
        self.is_recording = False
    def run(self):
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        while self.is_recording:
            with sr.Microphone() as source:
                print("Recording...") #Bắt đầu nhận diện giọng nói
                audio = recognizer.listen(source)
            try:
                self.audio_text = recognizer.recognize_google(audio) #đây là kết quả mà mô hình trả về
                print("Ket qua: ", self.audio_text) 
                self.create_video_from_text()
                self.audioTextChanged.emit(self.audio_text)
            except sr.UnknownValueError:
                print("Er!")  #Bỏ qua nếu không thể nhận diện
            except sr.RequestError as e:
                print(f"Lỗi: {e}")
>>>>>>> d2db99074efd303f87130d642d8077403ccdf2c1
    def start_recording(self):
        self.is_recording = True #Em đánh dấu cho biến is_recording là đang hoạt động
        self.start() 

    def stop_recording(self):
        self.is_recording = False
        self.wait()
        
    # hàm thêm đường dẫn ảnh từ văn bản nhận diện được
    def create_video_from_text(self):
<<<<<<< HEAD
        print("in create_video_from_text")
        print(self.audio_text)
        img_list = []
=======
        img_list = []
        img_list2 = []
>>>>>>> d2db99074efd303f87130d642d8077403ccdf2c1
        for char in self.audio_text.lower():
            if char != ' ':
                img_path = os.path.join(self.img_dir, f"{char}.jpg").replace('\\', '/')
                if os.path.exists(img_path):
                    img_list.append(img_path)
            elif char == ' ':
                img_path = os.path.join(self.img_dir, 'space.jpg').replace('\\', '/')
                if os.path.exists(img_path):
                    img_list.append(img_path)
            else:
                continue
<<<<<<< HEAD
        print("Image List:", img_list)
        if img_list:
            self.show_video(img_list)
=======
        for char in self.audio_text.lower():
            if char != ' ':
                img_path = os.path.join(self.img_dir2, f"{char}.jpg").replace('\\', '/')
                if os.path.exists(img_path):
                    img_list2.append(img_path)
            elif char == ' ':
                img_path = os.path.join(self.img_dir2, 'space.jpg').replace('\\', '/')
                if os.path.exists(img_path):
                    img_list2.append(img_path)
            else:
                continue
        print("Image List:", img_list)
        if img_list:
            self.show_video(img_list)
            self.show_video2(img_list2)
>>>>>>> d2db99074efd303f87130d642d8077403ccdf2c1
            self.audioTextChanged.emit("Video created!")
            print("Done")
    #tạo video bằng ảnh ngôn ngữ ký hiệu
    def show_video(self, img_list):
        frame_list = []
        for img_path in img_list:
            frame = cv2.imread(img_path)
            if frame is not None:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_list.append(rgb_image)

        if frame_list:
            fps = 0.25
            clip = ImageSequenceClip(frame_list, fps=fps)
            clip.write_videofile(self.video_output_path, codec='libx264', fps=fps)
<<<<<<< HEAD
class Video(QThread):
    vid = pyqtSignal(QImage)
    def run(self):
        video_data = b''
        self.hilo_corriendo = True
        video_path = r"output_video.mp4"
=======
    #tạo video bằng ảnh chữ cái thường
    def show_video2(self, img_list2):
        frame_list = []
        for img_path in img_list2:
            frame = cv2.imread(img_path)
            if frame is not None:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_list.append(rgb_image)

        if frame_list:
            fps = 0.25
            clip = ImageSequenceClip(frame_list, fps=fps)
            clip.write_videofile(self.video_output_path2, codec='libx264', fps=fps)
class Video(QThread):
    vid = pyqtSignal(QImage)
    def run(self):
        self.hilo_corriendo = True
        video_path = r"D:\a\output_video.mp4"
>>>>>>> d2db99074efd303f87130d642d8077403ccdf2c1
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frame rate
        delay = int(1000 / fps)  # Calculate delay between frames
        while self.hilo_corriendo:
            ret, frame = cap.read()
            if ret:
<<<<<<< HEAD
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = base64.b64encode(buffer)
                
                # Thêm kích thước của frame vào đầu chuỗi
                frame_size = len(frame_bytes).to_bytes(4, byteorder='big')
                video_data += frame_size + frame_bytes
        cap.release()
        client.send(video_data.encode('utf-8'))
        while self.hilo_corriendo:
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = base64.b64encode(buffer)
                
                # Thêm kích thước của frame vào đầu chuỗi
                frame_size = len(frame_bytes).to_bytes(4, byteorder='big')
                video_data += frame_size + frame_bytes
                
=======
>>>>>>> d2db99074efd303f87130d642d8077403ccdf2c1
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_Qt_format.scaled(890, 440, Qt.KeepAspectRatio)
                self.vid.emit(p)
<<<<<<< HEAD
                self.msleep(delay)  
=======
                
                self.msleep(delay)  # Introduce delay to match the frame rate
>>>>>>> d2db99074efd303f87130d642d8077403ccdf2c1
        cap.release()
    def stop(self):
        self.hilo_corriendo = False
        self.quit()
class Video2(QThread):
    vid2 = pyqtSignal(QImage)

    def run(self):
<<<<<<< HEAD
        buffer_size = 4096
        data = b''

        while True:
            while len(data) < 4:
                packet = client.recv(buffer_size)
                if not packet:
                    return
                data += packet

            frame_size = int.from_bytes(data[:4], byteorder='big')
            data = data[4:]

            while len(data) < frame_size:
                packet = client.recv(buffer_size)
                if not packet:
                    return
                data += packet

            frame_bytes = data[:frame_size]
            data = data[frame_size:]

            frame = base64.b64decode(frame_bytes)
            np_frame = np.frombuffer(frame, dtype=np.uint8)
            img = cv2.imdecode(np_frame, cv2.IMREAD_COLOR)
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(890, 440, Qt.KeepAspectRatio)
            self.vid2.emit(p)
            
            fps = 30
            frame_duration = 1.0 / fps
            time.sleep(frame_duration)
=======
        self.check = True
        video_path = r"D:\a\output_video2.mp4"
        cap = cv2.VideoCapture(video_path)
        
        fps = cap.get(cv2.CAP_PROP_FPS) 
        delay = int(1000 / fps)
        
        while self.check:
            ret, frame = cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_Qt_format.scaled(890, 440, Qt.KeepAspectRatio)
                self.vid2.emit(p)
                
                self.msleep(delay)  # Introduce delay to match the frame rate
        cap.release()
>>>>>>> d2db99074efd303f87130d642d8077403ccdf2c1

    def stop(self):
        self.check = False
        self.quit()
class Ham_Camera(QThread):
    luongPixMap1 = pyqtSignal(QImage)
    luongPixMap2 = pyqtSignal(QImage)
    luongString1 = pyqtSignal(str)
    luongString2 = pyqtSignal(str)
    luongClearSignal = pyqtSignal()
    checkTrungChanged = pyqtSignal(str)

    def __init__(self):
        super(Ham_Camera, self).__init__()
        self.checkTrung = ""
        self.trangThai = True
        self.string = ""
        self.string2 = ""
        self.frame_count_threshold = 20  # Số frame tối thiểu để hiển thị classname
        self.current_frame_count = 0
<<<<<<< HEAD
        self.luongString1.connect(self.update_string1)
        self.luongString2.connect(self.update_string2)
        self.luongClearSignal.connect(self.clear_string)
        self.num_of_timesteps = 7
        self.lm_list = []
        self.model = load_model(f'model_7.h5')

    def update_string1(self, new_string):
        self.string = new_string

    def update_string2(self, new_string):
        self.string2 = new_string

    def clear_string(self):
        self.string = ""

    @staticmethod
    def make_landmark_timestep(hand_landmarks):
        lm_list = []
        landmarks = hand_landmarks.landmark
        
        base_x = landmarks[0].x
        base_y = landmarks[0].y
        base_z = landmarks[0].z
        
        center_x = np.mean([lm.x for lm in landmarks])
        center_y = np.mean([lm.y for lm in landmarks])
        center_z = np.mean([lm.z for lm in landmarks])

        distances = [np.sqrt((lm.x - center_x)**2 + (lm.y - center_y)**2 + (lm.z - center_z)**2) for lm in landmarks[1:]]

        scale_factors = [1.0 / dist for dist in distances]

        lm_list.extend([0.0, 0.0, 0.0, landmarks[0].visibility])

        for lm, scale_factor in zip(landmarks[1:], scale_factors):
            lm_list.append((lm.x - base_x) * scale_factor)
            lm_list.append((lm.y - base_y) * scale_factor)
            lm_list.append((lm.z - base_z) * scale_factor)
            lm_list.append(lm.visibility)
        
        return lm_list
    @staticmethod
    def draw_landmark_on_image(results, img):
        h, w, _= img.shape
        bbox = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mpDraw.draw_landmarks(img, hand_landmarks, mphands.HAND_CONNECTIONS)
                for id, lm in enumerate(hand_landmarks.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    bbox.append([cx, cy])
            if bbox:
                x_min, y_min = np.min(bbox, axis=0)
                x_max, y_max = np.max(bbox, axis=0)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        return img
    @staticmethod
    def draw_class_on_image(label, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (20, 50)
        fontScale = 1
        fontColor = (0, 255, 0)
        thickness = 2
        lineType = 2
        cv2.putText(img, label,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
        return img

    def detect(self, lm_list):
        lm_list = np.array(lm_list)
        lm_list = np.expand_dims(lm_list, axis=0)
        results = self.model.predict(lm_list)
        predicted_label_index = np.argmax(results, axis=1)[0]
        classes = ['a', 'b', 'c']
        confidence = np.max(results, axis=1)[0]
        if confidence > 0.95:
            label = classes[predicted_label_index]
        else:
            label = "neutral"
        return label

    def process_frame(self, frame):
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                lm = self.make_landmark_timestep(hand_landmarks)
                self.lm_list.append(lm)
                if len(self.lm_list) == self.num_of_timesteps:
                    detect_thread = threading.Thread(target=self.run_detect, args=(self.lm_list,))
                    detect_thread.start()
                    self.lm_list = []
            frame = self.draw_landmark_on_image(results, frame)
        frame = self.draw_class_on_image(self.string, frame)
        return frame

    def run_detect(self, lm_list):
        label = self.detect(lm_list)
        if label != "neutral" and label != self.checkTrung:
            if label == "space":
                self.string += " "
            else:
                self.string += label
            self.luongString1.emit(self.string)
            self.checkTrung = label
            self.checkTrungChanged.emit(self.checkTrung)

    def run(self):
        cap = cv2.VideoCapture(0)

        while self.trangThai:
            ret, frame1 = cap.read()
            if not ret:
                break
            
            if ret:
                frame1 = self.process_frame(frame1)

                h, w, ch = frame1.shape
                bytes_per_line = ch * w
                convert_to_Qt_format = QImage(frame1.data, w, h, bytes_per_line, QImage.Format_BGR888)
                p = convert_to_Qt_format.scaled(891, 461, Qt.KeepAspectRatio)
                self.luongPixMap1.emit(p)
            else:
                break
        cap.release()

    def stop(self):
        self.trangThai = False

=======
        # Kết nối tín hiệu luongString1 của luồng camera với hàm update_string
        self.luongString1.connect(self.update_string1)
        self.luongString2.connect(self.update_string2)
        # Kết nối tín hiệu luongClearSignal của luồng camera với hàm clear_string
        self.luongClearSignal.connect(self.clear_string)

    def update_string1(self, new_string):
        self.string = new_string
    def update_string2(self, new_string):
        self.string2 = new_string
    def clear_string(self):
        # Xử lý khi nút "clear" được nhấn
        # Cập nhật giá trị của self.string thành chuỗi rỗng
        self.string = ""
    def run(self):
        # message_chat = client.recv(1024).decode('utf-8')
        with open('body_language.pkl', 'rb') as f:
            model = pickle.load(f)
        # server_ip = "26.23.20.235"
        # encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        # client = imagiz.Client("cc1", server_ip=server_ip)
        cap = cv2.VideoCapture(0) #khởi tạo webcam
        cap.set(640,640)
        # image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        x_ = []
        y_ = []
        
        with mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2) as holistic:
            while self.trangThai:# chạy liên tục quá trình nhận diện
                ret, frame1 = cap.read() 
                H, W, _ = frame1.shape
                if ret: #nếu như camera được khởi tạo thành công thì sẽ chạy phần xử lý, nếu không thì sẽ thoát chương trình
                    image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                    image1.flags.writeable = False
                    results1 = holistic.process(image1)
                    image1.flags.writeable = True
                    # cv2.imshow('r', image2)
                    mp_drawing.draw_landmarks(image1, results1.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                )
                    # mp_drawing.draw_landmarks(image2, results2.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                    #              mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                    #              mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                    #             )
                    try:
                        rh1 = results1.right_hand_landmarks.landmark
                        # print('rh1: ',rh1,'\n')
                        rh_row1 = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in rh1]).flatten())
                        # print(len(rh_row1))
                        row1 = rh_row1
                        print(type(rh_row1))
                        X1 = pd.DataFrame([row1])
                        # print('X1: ',X1,'\n')
                        with open("dat.txt", 'w') as f: 
                            rh1_str = ", ".join([str(landmark) for landmark in rh1]) 
                            f.write('rh1: ' + rh1_str + '\n\n')

                            # Chuyển đổi rh_row1 thành chuỗi
                            rh_row1_str = ", ".join(map(str, rh_row1))
                            f.write('rh_row1: ' + rh_row1_str + '\n\n')

                            # Chuyển đổi X1 thành chuỗi
                            X1_str = X1.to_string()  # Sử dụng to_string() để chuyển DataFrame thành chuỗi
                            f.write('X1: ' + X1_str + '\n')

                        body_language_class1 = model.predict(X1)[0]
                        print(model.predict(X1))
                        print(model.predict(X1)[0])
                        body_language_prob1 = model.predict_proba(X1)[0]
                        print(model.predict_proba(X1))
                        print(model.predict_proba(X1)[0])
                        if results1.right_hand_landmarks:
                            bbox1 = self.get_hand_bbox(results1.right_hand_landmarks, W, H)
                            cv2.rectangle(image1, bbox1[0], bbox1[1], (255, 255, 255), 2)
                            class_name1 = body_language_class1.split(' ')[0]
                            prob_text1 = f'{class_name1}: {round(body_language_prob1[np.argmax(body_language_prob1)], 2)}'
                            cv2.putText(image1, prob_text1, (bbox1[0][0], bbox1[0][1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)           
                        if body_language_prob1[np.argmax(body_language_prob1)] >= 0.85 and body_language_class1 != self.checkTrung:
                            if body_language_class1 == "space": 
                                self.string += " "
                                self.luongString1.emit(self.string)
                                self.checkTrung = body_language_class1
                                self.checkTrungChanged.emit(self.checkTrung)
                            else:
                                self.string += body_language_class1
                                self.luongString1.emit(self.string)
                                self.checkTrung = body_language_class1
                                self.checkTrungChanged.emit(self.checkTrung)
                        # image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
                        # image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                        

                        # rh2 = results2.right_hand_landmarks.landmark
                        # rh_row2 = list(np.array([[landmark.x, landmark.y, landmark.z] for landmark in rh2]).flatten())
                        # row2 = rh_row2
                        # X2 = pd.DataFrame([row2])
                        # body_language_class2 = model.predict(X2)[0]
                        # body_language_prob2 = model.predict_proba(X2)[0]
                        # if results2.right_hand_landmarks:
                        #     bbox2 = self.get_hand_bbox(results2.right_hand_landmarks, W2, H2)
                        #     cv2.rectangle(image2, bbox2[0], bbox2[1], (255, 255, 255), 2)
                        #     class_name2 = body_language_class2.split(' ')[0]
                        #     prob_text2 = f'{class_name2}: {round(body_language_prob2[np.argmax(body_language_prob2)], 2)}'
                        #     cv2.putText(image2, prob_text2, (bbox2[0][0], bbox2[0][1] - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                        # if body_language_prob2[np.argmax(body_language_prob2)] >= 0.85 and body_language_class2 != self.checkTrung2:
                        #     if body_language_class2 == "space":
                        #         self.string2 += " "
                        #         self.luongString2.emit(self.string2)
                        #         self.checkTrung2 = body_language_class2
                        #     else:
                        #         self.string2 += body_language_class2
                        #         self.luongString2.emit(self.string2)
                        #         self.checkTrung2 = body_language_class2
                    except:
                        pass
                    h, w, ch = image1.shape
                    bytes_per_line = ch * w
                    convert_to_Qt_format = QImage(image1.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    p = convert_to_Qt_format.scaled(891, 461, Qt.KeepAspectRatio)
                    self.luongPixMap1.emit(p)
                else:
                    break
        cap.release()
    def stop(self): 
        self.trangThai = False
>>>>>>> d2db99074efd303f87130d642d8077403ccdf2c1
    def get_hand_bbox(self, landmarks, image_width, image_height):
        x_min, x_max, y_min, y_max = float('inf'), 0, float('inf'), 0

        for landmark in landmarks.landmark:
            x, y = int(landmark.x * image_width), int(landmark.y * image_height)
            x_min = min(x_min, x) - 3
            x_max = max(x_max, x) + 1
            y_min = min(y_min, y) - 3
            y_max = max(y_max, y) + 1

        bbox = ((x_min, y_min), (x_max, y_max))
        return bbox

"""
Ham_Camera được sử dụng để khởi tạo webcam và chạy mô hình dự đoán của dự án, hình ảnh được ghi nhận từ webcam sẽ được
chuyển thành hình ảnh sau đó cập nhật lên label_cam, tốc dộ cập nhật gần như bằng với thời gian thực
"""
class Ham_Chinh(QMainWindow):
    messageSent = pyqtSignal(str)
    # Lớp Ham_Chinh là lớp chính của chương trình, chịu trách nhiệm khởi tạo các thành phần giao diện và kết nối các tín hiệu giữa các lớp.
    def __init__(self):
        # Gọi hàm khởi tạo của lớp QMainWindow
        super(Ham_Chinh, self).__init__()
        # Tải giao diện từ file ui.ui
        loadUi('main.ui', self)
        
        # Khởi tạo luồng camera
        self.Work = Video()
        self.Work2 = Video2()
        self.thread_camera = Ham_Camera()
        self.thread_camera.luongClearSignal.connect(self.process_string)
        self.thread_camera.checkTrungChanged.connect(self.handle_check_trung_changed)
        # Khởi tạo luồng video
<<<<<<< HEAD
        self.img_dir = r'C:\Users\chojl\Desktop\app\img1'
=======
        self.img_dir = r'D:\a\img'
>>>>>>> d2db99074efd303f87130d642d8077403ccdf2c1
        self.video_output_path = r'output_video.mp4'
        self.thread_vid = SpeechToVideoThread(self.img_dir, self.video_output_path)
        # Kết nối tín hiệu luongPixMap của luồng camera với hàm setCamera
        self.thread_camera.luongPixMap1.connect(self.setCamera1)
        self.thread_camera.luongPixMap2.connect(self.setCamera2)
        # Kết nối tín hiệu startcam của nút startcam với hàm khoiDongCamera
        self.startcam.clicked.connect(self.khoiDongCamera)
        # Kết nối tín hiệu pausecam của nút pausecam với hàm tamDungCamera
        self.pausecam.clicked.connect(self.tamDungCamera)
        # Kết nối tín hiệu clear của nút clear với hàm xoaToanBo
        self.clear.clicked.connect(self.xoaToanBo)
        # Kết nối tín hiệu delete_2 của nút delete_2 với hàm xoaChu
        self.delete_2.clicked.connect(self.xoaChu)
        self.space.clicked.connect(self.spacee)
        self.check.clicked.connect(self.checkk)
        self.send.clicked.connect(self.sendMess)
        self.messageSent.connect(self.send_message)
        #Kết nối tín hiệu speak với hàm nói ra văn bản
        # message = client.recv(1024).decode('utf-8')
        # self.text2.setText(message)
        # Kết nối tín hiệu luongString1 của luồng camera với hàm setText của label text
        self.thread_camera.luongString1.connect(self.text1.setText)
        self.thread_camera.luongString2.connect(self.text2.setText)
        #voice to text/video
        self.record_button.clicked.connect(self.start_recording)
        self.stop_record_button.clicked.connect(self.stop_recording)
        self.stop_record_button.setEnabled(False)
        self.play_video.clicked.connect(self.start_video)
        self.stop_video.clicked.connect(self.stop_vide)
        self.thread_vid.audioTextChanged.connect(self.text_2.setText)
        self.listen_thread = threading.Thread(target=self.listen_for_messages)
        self.listen_thread.start()
    def start_video(self):
        self.Work.start()
        self.Work2.start()
        self.Work.vid.connect(self.Imageupd_slot)
        self.Work2.vid2.connect(self.vidletter)
    def listen_for_messages(self):
        while True:
            try:
                message = client.recv(1024).decode('utf-8')
                print(f"Received message: {message}")
                if "OTHER_USER_IP:" in message:
                    other_user_ip = message.split(":")[1]
                    # subprocess.Popen(["python", "mainlstm.py"])
                    # time.sleep(7)
                    subprocess.Popen(["python", "client_camera.py"])
                    print("done")
                else:
                    self.text2.setText(message)
            except Exception as e:
                print("An error occurred!", e)
                client.close()
                break

    def sendMess(self):
        mess = self.text1.text()
        self.messageSent.emit(mess)

    def send_message(self, message):
        client.send(message.encode('utf-8'))
        print("Sent message successfully")
    def Imageupd_slot(self, Image):
        self.img_label.setPixmap(QPixmap.fromImage(Image))
    def vidletter(self, Image):
        self.img_label_2.setPixmap(QPixmap.fromImage(Image))
    def stop_vide(self):
        self.Work.stop()
        self.Work2.stop()
    def setCamera1(self, image1):
        # Cập nhật hình ảnh lên label cam
        self.camera1.setPixmap(QPixmap.fromImage(image1))
    def setCamera2(self, image2):
        # Cập nhật hình ảnh lên label cam
        self.camera2.setPixmap(QPixmap.fromImage(image2))
    def khoiDongCamera(self):
        # Khởi động luồng camera để bắt đầu nhận diện vật thể
        self.thread_camera.start()
    def tamDungCamera(self):
        # Dừng luồng camera để tạm dừng nhận diện vật thể
        self.thread_camera.stop()
        # Chờ luồng camera hoàn toàn dừng trước khi tiếp tục
        self.thread_camera.wait()
    def xoaToanBo(self):
        # Xóa toàn bộ nội dung trong label text
        self.thread_camera.luongClearSignal.emit()
        self.text1.setText()
    def process_string(self):
        # Truy cập và xử lý giá trị từ Ham_Camera
        self.thread_camera.string = ""  
        # Cập nhật giá trị trong Ham_Camera
        self.thread_camera.luongString1.emit(self.thread_camera.string)
        self.text1.setText()
    def xoaChu(self):
        # Xóa ký tự cuối cùng trong textt
        textt = self.text1.text()  
        textt = textt[:-1]
        print(textt)
        # Cập nhật textt lên label text
        self.text1.setText(textt)
        self.thread_camera.luongString1.emit(textt)
    def spacee(self):
        # Xóa ký tự cuối cùng trong textt
        textt = self.text1.text()  
        textt = textt + " "
        print(textt)
        # Cập nhật textt lên label text
        self.text1.setText(textt)
        self.thread_camera.luongString1.emit(textt)
    def checkk(self):
        self.thread_camera.checkTrung = ""
        self.thread_camera.checkTrungChanged.emit(self.thread_camera.checkTrung)
    def handle_check_trung_changed(self, new_check_trung):
        if new_check_trung == "":
            print("checkTrung done")
    def start_recording(self):
        self.record_button.setEnabled(False)
        self.stop_record_button.setEnabled(True)
        self.thread_vid.start_recording()
    def stop_recording(self):
        self.record_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)
        self.thread_vid.stop_recording()
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Ham_Chinh()
    window.setWindowTitle('MainApp')
    window.show()
    sys.exit(app.exec_())
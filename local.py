from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from keras.models import load_model
import cv2
import mediapipe as mp
import subprocess

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')

num_of_timesteps = 12
model = load_model('best_model_12.h5')

classes = ['a', 'b', 'c', 'o', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
         'l', 'm', 'n', 'p', 'q', 'r', 's', 'space', 't', 'u',
         'v', 'w', 'x', 'y', 'z', 'yes', 'no', 'me', 'you', 'hello',
         'i_love_you', 'thank_you', 'sorry', 'do', 'eat', 'what', 'why', 
         'who', 'where', 'how_much', 'go', 'happy', 'sad', 'bad']

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

NUM_HAND_LANDMARKS = 21

def process_frame(frame):
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        lm_list = make_dat(hand_landmarks)
        return lm_list
    return None

def make_dat(hand_landmarks):
    lm_list = []
    
    if hand_landmarks:
        hand_lm = hand_landmarks.landmark
        base_x = hand_lm[0].x
        base_y = hand_lm[0].y
        base_z = hand_lm[0].z
        center_x = np.mean([lm.x for lm in hand_lm])
        center_y = np.mean([lm.y for lm in hand_lm])
        center_z = np.mean([lm.z for lm in hand_lm])

        distances = [np.sqrt((lm.x - center_x)**2 + (lm.y - center_y)**2 + (lm.z - center_z)**2) for lm in hand_lm[1:]]
        scale_factors = [1.0 / dist for dist in distances]

        lm_list.append(0.0)
        lm_list.append(0.0)
        lm_list.append(0.0)
        lm_list.append(hand_lm[0].visibility)

        for lm, scale_factor in zip(hand_lm[1:], scale_factors):
            lm_list.append((lm.x - base_x) * scale_factor)
            lm_list.append((lm.y - base_y) * scale_factor)
            lm_list.append((lm.z - base_z) * scale_factor)
            lm_list.append(lm.visibility)
    else:
        lm_list.extend([0.0] * (NUM_HAND_LANDMARKS * 4))
    
    return lm_list

label = "Unknown"
confidence = 0

def detect(landmarks_list):
    global label
    global confidence
    
    landmarks_array = np.array(landmarks_list)
    landmarks_array = np.expand_dims(landmarks_array, axis=0)
    results = model.predict(landmarks_array)
    predicted_label_index = np.argmax(results, axis=1)[0]
    
    classes = ['a', 'b', 'c', 'o', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
         'l', 'm', 'n', 'p', 'q', 'r', 's', 'space', 't', 'u',
         'v', 'w', 'x', 'y', 'z', 'yes', 'no', 'me', 'you', 'hello',
         'i_love_you', 'thank_you', 'sorry', 'do', 'eat', 'what', 'why', 
         'who', 'where', 'how_much', 'go', 'happy', 'sad', 'bad']
    
    confidence = np.max(results, axis=1)[0]
    label = classes[predicted_label_index]
    
    print(f"Predicted label index: {predicted_label_index}")
    
    return label, confidence, predicted_label_index

@app.route('/api/analyze_frames', methods=['POST'])
def analyze_frames():
    try:
        print("Received request to analyze frames")
        print("Request headers:", request.headers)
        print("Request form:", request.form)
        print("Request files:", request.files)
        
        frames = [request.files[f'frame{i}'] for i in range(num_of_timesteps) if f'frame{i}' in request.files]
        print(f"Received {len(frames)} frames")

        if len(frames) != num_of_timesteps:
            return jsonify({'error': f'Expected {num_of_timesteps} frames, but received {len(frames)}'}), 400

        landmarks_list = [process_frame(cv2.imdecode(np.frombuffer(frame.read(), np.uint8), cv2.IMREAD_COLOR)) for frame in frames]
        valid_landmarks = [lm for lm in landmarks_list if lm is not None]
        print(f"Processed {len(valid_landmarks)} valid frames")

        if len(valid_landmarks) == num_of_timesteps:
            label, confidence, predicted_label_index = detect(valid_landmarks)
            response = {
                'label': label,
                'confidence': float(confidence),
                'predicted_label_index': int(predicted_label_index)
            }
            return jsonify(response)
        else:
            print(f"Not enough valid frames. Expected {num_of_timesteps}, got {len(valid_landmarks)}")
            return jsonify({'error': 'Not enough valid frames'}), 400
    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host= '0.0.0.0', port=5001)

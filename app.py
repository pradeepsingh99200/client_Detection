from flask import Flask, render_template, request, jsonify
import cv2 as cv
import mediapipe as mp
import numpy as np
import math
import time 
import base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Variables for blink detection
CEF_COUNTER = 0
TOTAL_BLINKS = 0
CLOSED_EYES_FRAME = 3  
start_time = None

# Euclidean distance function to calculate eye aspect ratio
def euclideanDistance(point1, point2):
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# Blinking Ratio function
def blinkRatio(landmarks):
    right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    # Right eye
    rh_right = landmarks[right_eye_indices[0]]
    rh_left = landmarks[right_eye_indices[8]]
    rv_top = landmarks[right_eye_indices[12]]
    rv_bottom = landmarks[right_eye_indices[4]]

    # Left eye
    lh_right = landmarks[left_eye_indices[0]]
    lh_left = landmarks[left_eye_indices[8]]
    lv_top = landmarks[left_eye_indices[12]]
    lv_bottom = landmarks[left_eye_indices[4]]

    rhDistance = euclideanDistance(rh_right, rh_left)
    rvDistance = euclideanDistance(rv_top, rv_bottom)
    lvDistance = euclideanDistance(lv_top, lv_bottom)
    lhDistance = euclideanDistance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/camera', methods=['POST'])
def camera():
    global TOTAL_BLINKS, CEF_COUNTER, start_time
    TOTAL_BLINKS = 0  
    CEF_COUNTER = 0  
    start_time = time.time()
    
    # Get user's name from the form submission
    username = request.form.get('username')
    return render_template('camera.html', username=username)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    global TOTAL_BLINKS, CEF_COUNTER, start_time
    data = request.json
    base64_image = data['frame'].split(',')[1]  # Strip the header from base64

    if not base64_image:  # Check if the base64 string is empty
        return jsonify({'status': 'No image received', 'total_blinks': TOTAL_BLINKS, 'ratio': None})

    frame = decode_base64_image(base64_image)

    if frame is None:  # Check if the frame is valid
        return jsonify({'status': 'Invalid image', 'total_blinks': TOTAL_BLINKS, 'ratio': None})

    # Convert the frame to RGB for Mediapipe
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    blink_status = "No blink detected"
    blink_ratio = None  # Initialize blink ratio

    if results.multi_face_landmarks:
        landmarks = [(int(pt.x * frame.shape[1]), int(pt.y * frame.shape[0])) for pt in results.multi_face_landmarks[0].landmark]
        blink_ratio = blinkRatio(landmarks)

        # Blink detection logic
        if blink_ratio > 5.5:  # Eye is considered closed
            CEF_COUNTER += 1
            blink_status = "Eyes closed"
        else:  # Eye is considered open
            if CEF_COUNTER >= CLOSED_EYES_FRAME:
                TOTAL_BLINKS += 1
                CEF_COUNTER = 0  # Reset after detecting a blink
            blink_status = "Eyes open"

    # Check if 30 seconds have passed
    elapsed_time = time.time() - start_time
    if elapsed_time > 30:
        return jsonify({'redirect': True, 'total_blinks': TOTAL_BLINKS})

    return jsonify({'status': blink_status, 'total_blinks': TOTAL_BLINKS, 'ratio': blink_ratio})
    
@app.route('/result/<int:total_blinks>')
def result(total_blinks):
    health_status = "Your eyes are healthy" if total_blinks >= 7 else "Your eyes are unhealthy"
    return render_template('result.html', total_blinks=total_blinks, health_status=health_status)

def decode_base64_image(base64_string):
    img_data = base64.b64decode(base64_string)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv.imdecode(np_arr, cv.IMREAD_COLOR)
    return img if img is not None else None  # Return None if decoding fails

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

app = Flask(__name__)
    

from flask import Flask, render_template, request, redirect, url_for
import cv2
import face_recognition
import pickle
import os

app = Flask(__name__)

# Load known face encodings and names
if os.path.exists("known_faces.pkl"):
    with open("known_faces.pkl", "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings = []
    known_face_names = []

# Initialize video capture for IR camera
video_capture = cv2.VideoCapture('/dev/video2')

@app.route('/')
def index():
    return render_template('framework.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    print("Please position your face in front of the camera.")
    ret, frame = video_capture.read()
    if not ret:
        return "Failed to capture frame", 400

    rgb_frame = frame[:, :, ::-1]
    face_encodings = face_recognition.face_encodings(rgb_frame)
    if face_encodings:
        known_face_encodings.append(face_encodings[0])
        known_face_names.append(name)
        with open("known_faces.pkl", "wb") as f:
            pickle.dump((known_face_encodings, known_face_names), f)
        return redirect(url_for('index'))
    else:
        return "No face detected. Please try again.", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

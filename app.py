import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

from flask import Flask, render_template, request, redirect, url_for
import cv2
import dlib
import pickle
import time
import numpy as np

app = Flask(__name__)

# Initialize video capture for IR camera
video_capture = cv2.VideoCapture('/dev/video2')

# Ensure camera access
if not video_capture.isOpened():
    print("Error: Could not open video device.")
    exit()

# Load Dlib's face detector, shape predictor, and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load known face encodings and names
if os.path.exists("known_faces.pkl"):
    with open("known_faces.pkl", "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings = []
    known_face_names = []

@app.route('/')
def index():
    return render_template('framework.html')

@app.route('/register', methods=['POST'])
def register():
    name = request.form['name']
    
    # Capture frames for 2 seconds
    start_time = time.time()
    frames = []
    
    while time.time() - start_time < 2:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            return redirect(url_for('index'))
        frames.append(frame)
        
        # Display captured frame for debugging
        cv2.imshow('Captured Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  # Allow user to quit early
    
    cv2.destroyAllWindows()  # Close any open windows
    
    # Process captured frames to find faces
    for frame in frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        detections = detector(rgb_frame, 1)
        shapes = dlib.full_object_detections()
        
        for detection in detections:
            shape = predictor(rgb_frame, detection)
            shapes.append(shape)
        
        # Compute face descriptors
        if len(shapes) > 0:
            face_descriptors = [np.array(face_rec_model.compute_face_descriptor(rgb_frame, shape)) for shape in shapes]
            for face_descriptor in face_descriptors:
                known_face_encodings.append(face_descriptor)
                known_face_names.append(name)
            
            # Save the updated known faces
            with open("known_faces.pkl", "wb") as f:
                pickle.dump((known_face_encodings, known_face_names), f)
            
            print(f"Registered {name} successfully.")
            
            # Release the video capture device
            video_capture.release()
            
            return redirect(url_for('index'))
    
    print("No faces detected.")
    
    # Release the video capture device
    video_capture.release()
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



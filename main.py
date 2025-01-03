import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import dlib
import face_recognition
import pickle
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor

# Load known face encodings and names
if os.path.exists("known_faces.pkl"):
    with open("known_faces.pkl", "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings = []
    known_face_names = []

# Ensure encodings are lists
known_face_encodings = [list(encoding) for encoding in known_face_encodings]

# Initialize video capture for IR camera
video_capture = cv2.VideoCapture('/dev/video2')

# Load Dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Variables to keep track of identified names
identified_names = []

def process_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector(rgb_frame, 0)  # Use 0 for no up-sampling

    face_encodings = []
    face_locations = []

    # Use dlib.full_object_detections to store landmarks
    shapes = dlib.full_object_detections()
    for detection in detections:
        shape = predictor(rgb_frame, detection)
        shapes.append(shape)
        face_locations.append((detection.top(), detection.right(), detection.bottom(), detection.left()))

    # Compute face descriptors
    if len(shapes) > 0:
        face_descriptors = [face_rec_model.compute_face_descriptor(rgb_frame, shape) for shape in shapes]
        face_encodings = [np.array(descriptor) for descriptor in face_descriptors]

    results = []
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if known_face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        results.append((top, right, bottom, left, name))
    
    return results

def recognize_faces():
    frame_count = 0  # To keep track of the frames processed
    last_recognized_time = None  # To track the time of the last recognition
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            while True:
                start_time = time.time()
                
                ret, frame = video_capture.read()
                if not ret:
                    print("Failed to capture frame.")
                    break

                frame_count += 1

                if frame_count % 2 == 0:  # Process every 2nd frame for faster processing
                    future = executor.submit(process_frame, frame)
                    identified_names.clear()
                    identified_names.extend(future.result())

                # Display identified names on the frame
                for (top, right, bottom, left, name) in identified_names:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('InfraFace', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Closing application...")
                    break

                # Check if a face has been recognized
                if identified_names:
                    for _, _, _, _, name in identified_names:
                        if name != "Unknown":
                            print(f"Successfully recognized: {name}")
                            last_recognized_time = time.time()  # Update the last recognized time

                # Close the application 2 seconds after recognizing a face
                if last_recognized_time and time.time() - last_recognized_time > 2:
                    print("Closing application after successful recognition...")
                    break

                # Calculate the time taken for one frame processing and wait to maintain high FPS
                end_time = time.time()
                processing_time = end_time - start_time
                wait_time = max(0, (1 / 120) - processing_time)
                time.sleep(wait_time)

    finally:
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    recognize_faces()



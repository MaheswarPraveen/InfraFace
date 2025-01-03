import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import dlib
import pickle
import numpy as np

# Load known face encodings and names
if os.path.exists("known_faces.pkl"):
    with open("known_faces.pkl", "rb") as f:
        known_face_encodings, known_face_names = pickle.load(f)
else:
    known_face_encodings = []
    known_face_names = []

# Initialize video capture for IR camera
video_capture = cv2.VideoCapture('/dev/video2')

# Load Dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def recognize_faces():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        detections = detector(rgb_small_frame, 1)  # Use 1 for up-sampling

        face_encodings = []
        face_locations = []

        # Use dlib.full_object_detections to store landmarks
        shapes = dlib.full_object_detections()
        for detection in detections:
            shape = predictor(rgb_small_frame, detection)
            shapes.append(shape)
            face_locations.append((detection.top() * 2, detection.right() * 2, detection.bottom() * 2, detection.left() * 2))

        # Compute face descriptors
        if len(shapes) > 0:
            face_descriptors = face_rec_model.compute_face_descriptor(rgb_small_frame, shapes)
            face_encodings = [np.array(descriptor) for descriptor in face_descriptors]

        # Match faces with known encodings
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = dlib.chinese_whispers_clustering(known_face_encodings, 0.4)  # Adjust threshold as needed
            name = "Unknown"

            if known_face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('InfraFace', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    recognize_faces()


import cv2
import dlib
import face_recognition
import pickle
import numpy as np

# Load known face encodings and names
with open("/home/doraemon/infraface/known_faces.pkl", "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)

# Initialize video capture for IR camera
video_capture = cv2.VideoCapture('/dev/video2')

# Load Dlib's face detector and face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/doraemon/infraface/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("/home/doraemon/infraface/dlib_face_recognition_resnet_model_v1.dat")

def authenticate():
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to capture frame.")
        return False

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector(rgb_frame, 0)

    face_encodings = []
    face_locations = []
    shapes = dlib.full_object_detections()
    for detection in detections:
        shape = predictor(rgb_frame, detection)
        shapes.append(shape)
        face_locations.append((detection.top(), detection.right(), detection.bottom(), detection.left()))

    if len(shapes) > 0:
        face_descriptors = [face_rec_model.compute_face_descriptor(rgb_frame, shape) for shape in shapes]
        face_encodings = [np.array(descriptor) for descriptor in face_descriptors]

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        if any(matches):
            return True
    return False

if __name__ == "__main__":
    if authenticate():
        exit(0)
    else:
        exit(1)


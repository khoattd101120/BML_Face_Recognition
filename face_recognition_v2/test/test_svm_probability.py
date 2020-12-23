import face_recognition
import cv2
import numpy as np
from sklearn.externals import joblib
from tensorflow.keras.models import load_model
from src.modules.training_svm_model.face_embeddings import get_embedding
from src.modules.face_emotion.emotion_detection import EmotionDetection
import json
import dlib


# def isUnknown():
#     pass

svm_model = joblib.load(
    config.BASEPATH + 'trained_models/face_recognition_model/model.pkl')
facenet_model = load_model(
    config.BASEPATH + 'trained_modelsfacenet_embedding_model/facenet_keras.h5')

name_mapping = json.load(
    open("/home/nguyendhn/PycharmProjects/devfest_hackathon_2019/src/modules/integration/name_encoder_mapping.txt"))


link = "/home/nguyendhn/Pictures/Webcam/2019-10-11-001313.jpg"
image = cv2.imread(link)

small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
rgb_small_frame = small_frame[:, :, ::-1]

# Find all the faces and face encodings in the current frame of video
face_locations = face_recognition.face_locations(image, number_of_times_to_upsample=0, model="cnn")

top, right, bottom, left = face_locations[0]
only_face_rgb = image[top:bottom, left:right, :]
only_face_rgb = cv2.resize(only_face_rgb, (160, 160))

embedding = get_embedding(facenet_model, only_face_rgb)
probabilities = svm_model.predict_proba([embedding])[0]

name_pred = name_mapping[str(np.argmax(probabilities))]

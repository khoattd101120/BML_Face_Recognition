import face_recognition
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from src.modules.training_svm_model.face_embeddings import get_embedding
from src.modules.face_emotion.emotion_detection import EmotionDetection
from numpy import savez_compressed
import dlib
import json
import requests
from config import config
import datetime
import json

scale = 0.25

SVM_CLASSIFIER_THRESHOLD = 0.999999

sess = requests.session()

base_link = "http://34.227.31.49:3000/customer-coming"

headers = {"content-type": "application/json", "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/72.0.3626.119 Safari/537.36"}


svm_model = joblib.load(
    config.BASEPATH + 'trained_models/face_recognition_model/model.pkl')
facenet_model = load_model(
    config.BASEPATH + 'trained_models/facenet_embedding_model/facenet_keras.h5')
emotion_model = load_model(config.BASEPATH + 'trained_models/emotion_model/fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)

gender_model = load_model(config.BASEPATH + 'trained_models/gender_model/simple_CNN.81-0.96.hdf5', compile=False)

name_mapping = json.load(
    open(config.BASEPATH + "assets/name_encoder_mapping.txt"))

emotion_mapping = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}

# getting input model shapes for inference
emotion_target_size = emotion_model.input_shape[1:3]

gender_target_size = gender_model.input_shape[1:3]
print(gender_target_size)


# Initialize some variables
face_locations = []
face_names = []
face_feature_vectors = []

batch_frames = 20

recognition_detector = []
emotion_detector = []
gender_detector = []

current_name = ''
current_emotion = ''

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    out = frame.copy()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(out, (0, 0), fx=scale, fy=scale)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=0, model="cnn")

    # Just select the first face
    if len(face_locations) > 0:
        top, right, bottom, left = face_locations[0]
        only_face_rgb = rgb_small_frame[top:bottom, left:right, :]
        only_face_rgb = cv2.resize(only_face_rgb, (160, 160))

        embedding = get_embedding(facenet_model, only_face_rgb)
        probabilities = svm_model.predict_proba([embedding])[0]

        print("MAX", max(probabilities))

        if max(probabilities) < SVM_CLASSIFIER_THRESHOLD:
            name = "unknown"
        else:
            name = name_mapping[str(list(probabilities).index(max(probabilities)))]

        top *= int(1 / scale)
        right *= int(1 / scale)
        bottom *= int(1 / scale)
        left *= int(1 / scale)

        # Draw a box around the face or find the largest (TODO)
        cv2.rectangle(out, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(out, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(out, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Process for emotion detection
        only_face_gray = cv2.cvtColor(only_face_rgb, cv2.COLOR_BGR2GRAY)
        try:
            only_face_gray = cv2.resize(only_face_gray, emotion_target_size)
            only_face_gray = only_face_gray / 255.
            only_face_gray = np.expand_dims(only_face_gray, 0)
            only_face_gray = np.expand_dims(only_face_gray, -1)
            only_face_rgb = cv2.resize(only_face_rgb, gender_target_size)
        except:
            continue

        emotion_prediction = emotion_model.predict(only_face_gray)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_mapping[int(emotion_label_arg)]

        if emotion_text == 'angry':
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text == 'sad':
            color = emotion_probability * np.asarray((0, 0, 255))
        elif emotion_text == 'happy':
            color = emotion_probability * np.asarray((255, 255, 0))
        elif emotion_text == 'surprise':
            color = emotion_probability * np.asarray((0, 255, 255))
        else:
            color = emotion_probability * np.asarray((0, 255, 0))

        color = color.astype(int)
        color = color.tolist()

        only_face_rgb = np.expand_dims(only_face_rgb[:, :], 0)
        only_face_rgb = only_face_rgb / 255.
        gender_prediction = gender_model.predict(only_face_rgb)
        gender_label_arg = np.argmax(gender_prediction)
        if gender_label_arg == 0:
            gender_text = "female"
        else: # Assume that we have only 0 and 1 returned
            gender_text = "male"

        cv2.rectangle(out, (left, top + 35), (left, top), (255, 0, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(out, emotion_text + "=======" + gender_text, (left - 6, top - 6), font, 1.0, color, 1)

        if len(recognition_detector) == batch_frames:
            current_name = max(set(recognition_detector), key=recognition_detector.count)
            current_emotion = max(set(emotion_detector), key=emotion_detector.count)
            current_gender = max(set(gender_detector), key=gender_detector.count)

            if current_name == "unknown":
                data = {'id': None,
                        'gender': current_gender,
                        'emotion': current_emotion}

            else:
                data = {'id': current_name,
                        'gender': current_gender,
                        'emotion': current_emotion}

            r = sess.post(url=base_link, data=json.dumps(data), headers=headers)

            print("CURRENT NAME", current_name)

            recognition_detector = []
            emotion_detector = []
            face_feature_vectors = []
            gender_detector = []
        else:
            recognition_detector.append(name)
            emotion_detector.append(emotion_text)
            face_feature_vectors.append(embedding)
            gender_detector.append(gender_text)

    else:
        recognition_detector = []
        emotion_detector = []
        gender_detector = []

    # Display the resulting image

    winname = "AI Process"
    cv2.namedWindow(winname)  # Create a named window
    cv2.moveWindow(winname, 1300, 550)  # Move it to (40,30)
    cv2.imshow(winname, out)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()



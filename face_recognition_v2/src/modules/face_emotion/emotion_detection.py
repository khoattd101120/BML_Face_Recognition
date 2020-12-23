# from src.utils.inferences import *
import numpy as np
from keras.models import load_model
from config import config



emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}


class EmotionDetection:
    def __init__(self, emotion_model_path=config.BASEPATH + "trained_models/emotion_model/fer2013_mini_XCEPTION.102-0.66.hdf5"):
        self.model = load_model(emotion_model_path, compile=False)

    def _get_model(self):
        return self.model

    def predict(self, face_gray, v2=True):
        """
        Predict from face after gray scale
        Parameters:
        ==========
        :param face_gray: np.array(3D) with Gray color only
        :param v2:

        Results:
        =======
        :param output:
        """
        face_gray = face_gray.astype('float32')
        face_gray = face_gray / 255.0
        if v2:
            face_gray = face_gray - 0.5
            face_gray = face_gray * 2.0
        face_gray = np.expand_dims(face_gray, 0)
        face_gray = np.expand_dims(face_gray, -1)
        emotion_prediction = self.model.predict(face_gray)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        output = emotion_text
        probability = emotion_probability
        return output, probability





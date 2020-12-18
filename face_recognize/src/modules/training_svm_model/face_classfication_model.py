# develop a classifier for the 5 Celebrity Faces Dataset
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import matplotlib.pyplot   as  plt
import numpy as np
import warnings
import pickle
from sklearn.externals import joblib
from config import config


warnings.filterwarnings("ignore")


def train_SVM(data):
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print('Dataset: train=%d, test=%d' % (trainX.shape[0], testX.shape[0]))

    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)

    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)

    name_mapping = dict(zip(out_encoder.classes_, out_encoder.transform(out_encoder.classes_)))
    print(name_mapping)

    name_mapping = {int(value): key for key, value in name_mapping.items()}

    import json
    json.dump(name_mapping, open(
        config.BASEPATH + 'assets/name_encoder_mapping.txt', 'w'))
    testy = out_encoder.transform(testy)

    # fit model
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
    # Save the trained model as a pickle string. 
    joblib.dump(model, config.BASEPATH + 'trained_models/face_recognition_model/model.pkl')
    print("Load model !")

    # Load the model from the file 
    model = joblib.load(config.BASEPATH + 'trained_models/face_recognition_model/model.pkl')
    # predict
    yhat_train = model.predict(trainX)
    yhat_test = model.predict(testX)
    # score

    score_train = accuracy_score(trainy, yhat_train)
    score_test = accuracy_score(testy, yhat_test)
    # summarize
    print('Accuracy: train=%.3f, test=%.3f' % (score_train * 100, score_test * 100))

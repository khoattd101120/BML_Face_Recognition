from src.modules.training_svm_model.data_processing import load_faces, load_dataset
from src.modules.training_svm_model.face_embeddings import faces_embeddings
from numpy import savez_compressed
from src.modules.training_svm_model.face_classfication_model import train_SVM
from numpy import load
from config import config


if __name__ == '__main__':
    print("Start to load train dataset")
    # load train dataset
    trainX, trainy = load_dataset('/home/nguyendhn/PycharmProjects/data/train/')
    print(trainX.shape, trainy.shape)
    print("Start to test dataset")
    # load test dataset
    testX, testy = load_dataset('/home/nguyendhn/PycharmProjects/data/val/')
    # save arrays to one file in compressed format

    savez_compressed(config.BASEPATH + 'assets/faces-dataset.npz', trainX, trainy, testX, testy)
    print("Start to face embeddings")

    # load the face dataset
    data = load(config.BASEPATH + 'assets/faces-dataset.npz')

    faces_embeddings(data)
    print("Start to train")

    # load dataset
    data = load(config.BASEPATH + 'assets/faces-embeddings.npz')
    train_SVM(data)

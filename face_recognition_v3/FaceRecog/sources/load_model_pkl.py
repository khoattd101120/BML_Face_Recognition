import pickle

path = r'C:\Users\tinnvt\Documents\BasicML\Project\Face ' \
       r'Recognition\BML_Face_Recognition\Face_Recog\FaceRecog\Models\bml_vin_facenet.pkl '
data = pickle.load(open(path, "rb"))
print(data)
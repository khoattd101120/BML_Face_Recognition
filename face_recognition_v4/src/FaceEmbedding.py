import sys
import pickle
import cv2
import os
import glob


class FaceEmbedding:
    def __init__(self, face_model):
        self.face_model = face_model
        self.known_embeddings = []
        self.known_names = []
        self.root = os.getcwd()


    def embed_face(self, image):
        # Detect face
        input  = self.face_model.get_input(image)
        # print(len(bbox_face))
        # print(bbox_face[0].shape)
        if input is None:
            return input
        bbox, face = input
        # Get the face embedding vector
        face_embedding = self.face_model.get_feature(face)
        return bbox, face_embedding
    def embed_faces(self, img_paths, save = None, embedding_path = None):
        """
        :param img_path:
        :type img_path:
        :param save:
            default None: not save
            1 if replace old version
            2 if update embedding
        :type save:
        :param embedding_path:
        :type embedding_path:
        :return:
        :rtype:
        """
        l = len(img_paths)
        total = 0
        for (i, img_path) in enumerate(img_paths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(i + 1, l))

            name = img_path.split(os.path.sep)[-2]
            print(name)
            # load the image
            image = cv2.imread(img_path)

            embed_face = self.embed_face(image)
            if embed_face is None: #here
                print('-----------------------{}'.format(name))
                continue
            self.known_names.append(name)
            self.known_embeddings.append(embed_face[1])
            total += 1

        print(total, " faces embedded")

        data = {"embeddings": self.known_embeddings, "names": self.known_names}
        if save is None:
            return data
        else:
            if embedding_path is None:
                raise Exception("No embedding_path specific!")
            embedding_path = os.path.join(self.root, '../src', embedding_path)
            if save == 1: # replace
                self.save_pickle(data, embedding_path)
            elif save == 2:
                if not os.path.exists(embedding_path):
                    print(self.root)
                    print(__name__, embedding_path)
                    print(__name__, 'save 2, path not exists')
                    self.save_pickle(data, embedding_path)
                else:
                    data = self.load_pickle(embedding_path)
                    data['embeddings'].extend(self.known_embeddings)
                    data['names'].extend(self.known_names)
                    self.save_pickle(data, embedding_path)
        return data
    def save_pickle(self, data, embedding_path):
        embedding_path = os.path.join(self.root, '../src',embedding_path)
        print(embedding_path)
        f = open(embedding_path, "wb")
        f.write(pickle.dumps(data))
        f.close()
    def load_pickle(self,embedding_path):
        embedding_path = os.path.join(self.root, '../src', embedding_path)
        if not os.path.exists(embedding_path):
            raise Exception("Embedding_path not exists! \n {}".format(embedding_path))

        else:
            return pickle.loads(open(embedding_path, "rb").read())
if __name__ == '__main__':
    import sys

    sys.path.append('../insightface/deploy')
    sys.path.append('../insightface/src/common')
    import face_model
    from Namespace import Namespace
    from glob import glob

    args = Namespace(det=0, embeddings='../src/outputs/embeddings_duy.pickle', flip=0, ga_model='', gpu=0,
                          image_size='112,112', model='../insightface/models/model-y1-test2/model,0', threshold=1.24)
    embedding_model = face_model.FaceModel(args)
    face_embedding = FaceEmbedding(embedding_model)

    # Test embed one image
    # img = cv2.imread('../dataset/ameliezilber/Train/img_1.jpg')
    # embedding = face_embedding.embed_face(img)
    # print(embedding.shape)
    # print(embedding)

    # Test embed more image

    img_paths = glob('../dataset/*/*.*')
    print(img_paths)
    embeddings = face_embedding.embed_faces(img_paths, save = 1, embedding_path= 'outputs/embeddings.pickle')

    # Test cap image
    # face_embedding.start_capture('a')


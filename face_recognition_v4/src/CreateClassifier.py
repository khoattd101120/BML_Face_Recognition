import sys
sys.path.append('../insightface/deploy')
sys.path.append('../insightface/src/common')
import pickle
import face_model
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold

from Namespace import Namespace
from FaceEmbedding import FaceEmbedding


class CreateClassifier:
	def __init__(self, embedding_path = '../src/outputs/embeddings.pickle'):
		self.args = Namespace(det=0, embeddings='../src/outputs/embeddings.pickle', flip=0, ga_model='', gpu=0, image_size='112,112', model='../insightface/models/model-y1-test2/model,0', threshold=1.24)
		self.embedding_path = embedding_path
		self.face_model = face_model.FaceModel(self.args)
		self.face_embedding = FaceEmbedding(self.face_model)
		try:
			self.data = self.face_embedding.load_pickle(embedding_path)
			self.labels = self.data["names"]
			self.embeddings = np.array(self.data["embeddings"])
			self.num_classes = len(np.unique(self.labels))
			self.labels_unique = np.unique(self.labels).tolist()
			self.labels = [self.labels_unique.index(i) for i in self.labels]
		except:
			self.data = None
			self.labels = []
			self.embeddings = np.array([])
			self.num_classes = None
			self.labels_unique = np.array([])
			self.labels = []

		try:
			self.model = self.face_embedding.load_pickle(r'..\src\outputs\knn_model.pickle')
		except:
			self.model = None
	def update_data(self):
		self.data = self.face_embedding.load_pickle(self.embedding_path)
		self.labels = self.data["names"]
		self.embeddings = np.array(self.data["embeddings"])
		self.num_classes = len(np.unique(self.labels))
		self.labels_unique = np.unique(self.labels).tolist()
		self.labels = [self.labels_unique.index(i) for i in self.labels]
	def train(self, classifier = 'svm'):
		x_train, x_test, y_train, y_test = train_test_split(self.embeddings, self.labels, stratify= self.labels, test_size=0.3)
		print(x_train.shape)
		print(x_test.shape)
		if classifier == 'svm':
			model = SVC(C = 10000, gamma= 0.01, probability=True)
			model.fit(x_train, y_train)
			print('score', model.score(x_test, y_test))
		elif classifier == 'knn':
			model = KNeighborsClassifier(n_neighbors= 5)
			model.fit(x_train, y_train)
			print('score', model.score(x_test, y_test))
		self.face_embedding.save_pickle(model, 'outputs/{}_model.pickle'.format(classifier))
		self.model = model
		return model
	def test(self):
		img_path = [
			'https://ivcdn.vnecdn.net/giaitri/images/web/2020/12/21/mv-chung-ta-cua-hien-tai-1608516586.jpg?w=750&h=450&q=100&dpr=1&fit=crop&s=XvJG9jMhjzUhE752ufb9Jw',
			'https://ivcdn.vnecdn.net/giaitri/images/web/2020/12/21/mv-chung-ta-cua-hien-tai-1608516586.jpg?w=750&h=450&q=100&dpr=1&fit=crop&s=XvJG9jMhjzUhE752ufb9Jw',
			'https://icdn.dantri.com.vn/zoom/1200_630/2020/07/06/sontungmtp-5-1593996464465.jpeg',
			'https://cdn.tgdd.vn/Files/2020/07/06/1268110/4-_1600x900-800-resize.jpg',
			'https://cdn.vietnammoi.vn/2019/11/27/de-thi-ve-son-tung-mtp-1-15748482469981213385641.jpg',
			'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTIImePao59LOzoer7nBn5m9zID5457GQPNig&usqp=CAU',
			'https://avatar-ex-swe.nixcdn.com/singer/avatar/2019/07/17/d/e/0/2/1563332636822_600.jpg'

			# diem my 5
			'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTAeXecPQAtQs_hHt1sVB8JTvlsA0wqoo86Hw&usqp=CAU',
			'https://cdn.24h.com.vn/upload/1-2020/images/2020-01-02/1577976269-461-diem-my-9x-diem-my-9x-14-crop-15463198290651558635023-1577775800-width1000height888.jpg',
			'https://image2.baonghean.vn/w607/Uploaded/2020/tamzshuztnzm/2020_10_08/quakhubathaocuadiemmy9xdoihon100trieulamngucdapbatcomcuachong11602001112408width650height245_oujy.jpg',
			'https://img.giaoduc.net.vn/w700/Uploaded/2020/wpxlzdjwp/2012_06_21/hotgirl-diem-my-giaoduc.net%20(11).jpg',
			'https://photo-baomoi.zadn.vn/w700_r1/2020_03_27_296_34469540/8ed4ae13a7504e0e1741.jpg',
			# hoang yen chibi 10
			'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTSdQinu58Eg0q4WdB2F8zZ6ERD2okiUU3WTQ&usqp=CAU',
			'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRGb4EB6A-K-wxvYRkEcyWagW1-PO3CRbS6Ow&usqp=CAU',
			'https://cdn.pose.com.vn/assets/2018/07/hoangyenchibi812_31042993_1909903412373956_8276358643171983360_n.jpg',
			'https://photo-baomoi.zadn.vn/w700_r1/2020_12_18_119_37373467/dade2c0df34e1a10435f.jpg',
			'https://media-cdn.laodong.vn/Storage/NewsPortal/2018/9/13/630708/1522463265-246-Vua-T.jpg?w=414&h=276&crop=auto&scale=both',
			'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQjG0FZgH2_jxL2bZwBLYIf-X9xxtFuRBqeyg&usqp=CAU',
			'https://avatar-ex-swe.nixcdn.com/singer/cover/2020/01/06/0/b/0/d/1578286624991.jpg',
			'https://xone.fm/wp-content/uploads/2020/11/h0-10.png',
			# tran thanh
			'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTKsBqvGUPPI8HZQD34SmaYdgEoSXIzROfJlw&usqp=CAU'
			]

	def url_to_image(self, url):
		import urllib.request
		import cv2
		import numpy as np
		import urllib
		resp = urllib.request.urlopen(url)
		image = np.asarray(bytearray(resp.read()), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
		return image



# print(svm.score(x_test, y_test))

if __name__ == "__main__":
	create_classifier = CreateClassifier()
	# create_classifier.face_embedding.embed_faces()
	model = create_classifier.train(classifier='svm')

	img = create_classifier.url_to_image('https://znews-photo.zadn.vn/w660/Uploaded/qfssu/2020_12_20/Untitled.jpg')
	s = time.time()
	bbox, embedding = create_classifier.face_embedding.embed_face(img)
	print(create_classifier.labels_unique[model.predict([embedding])[0]])

	print( model.predict_proba([embedding])[0] )
	print('time:', time.time() - s)
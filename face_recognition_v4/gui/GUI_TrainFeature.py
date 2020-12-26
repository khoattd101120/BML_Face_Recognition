import os
import glob
def train_classifer(name, recognizer):

	name = name.lower().replace(' ', '_')
	data_dir = os.path.join(os.getcwd(), '../dataset_tmp/*/*.jpg')

	img_paths = glob.glob(data_dir)
	print(__name__, img_paths)
	recognizer.face_embedding.embed_faces(img_paths, save=2, embedding_path='outputs/embeddings.pickle')
	# recognizer.embedding_path = '../src/outputs/embeddings_duy_new_full_v3.pickle'
	recognizer.update_data()
	recognizer.train('knn')

	# Di chuyển ảnh về folder dataset
	data_dir_new = os.path.join(os.getcwd(), '../dataset')
	for data_dir in glob.glob(os.path.join(os.getcwd(), '../dataset_tmp/*')):
		# print(data_dir)
		os.system(f'move {data_dir} {data_dir_new}')


if __name__ == '__main__':
    train_classifer('duy', 'ahha')
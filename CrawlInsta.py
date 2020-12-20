import os
import glob
import dlib
import cv2
import re
import numpy as np
class CrawlInsta:
    def __init__(self):
        self.root = os.getcwd()
        self.path = os.path.join(self.root, 'images')
        self.face_detector = dlib.get_frontal_face_detector()
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        # change the current working directory
        os.chdir(self.path)
    def load_image(self, name, update = True):
        try:
            if update:
                os.system('instaloader --fast-update --no-video-thumbnails --no-videos --no-captions \
                --no-metadata-json --no-compress-json --count 100 {}'.format(name))
            else:
                os.system('instaloader --no-video-thumbnails --no-videos --no-captions \
                --no-metadata-json --no-compress-json --count 100 {} '.format(name))
        except:
            pass
        self.remove_redundant(name)
    def load_images(self, names, update = True):
        for name in names:
            self.load_image(name, update)
    def remove_redundant(self, folder):
        # rename path contain space to _
        # tmp1 = os.path.join(self.path, folder)
        # tmp2 = self.no_accent_vietnamese(tmp1).replace(' ', '_').replace('download', '')
        # os.rename(tmp1, tmp2)
        for redundant in glob.glob(os.path.join(self.path, folder, '*.*')):
            if not (redundant.endswith('.jpg') or redundant.endswith('.png')):
                os.remove(redundant)
            else: # remove image not contains face
                img = cv2.imread(redundant)
                # print(img)
                small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]
                faces = self.face_detector(rgb_small_frame, 1)
                print(len(faces))
                if len(faces) != 1:
                    os.remove(redundant)
        try:
            os.remove(os.path.join(self.path, folder, 'id'))
        except:
            pass
    def no_accent_vietnamese(self, s):
        s = re.sub('[áàảãạăắằẳẵặâấầẩẫậ]', 'a', s)
        s = re.sub('[ÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ]', 'A', s)
        s = re.sub('[éèẻẽẹêếềểễệ]', 'e', s)
        s = re.sub('[ÉÈẺẼẸÊẾỀỂỄỆ]', 'E', s)
        s = re.sub('[óòỏõọôốồổỗộơớờởỡợ]', 'o', s)
        s = re.sub('[ÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ]', 'O', s)
        s = re.sub('[íìỉĩị]', 'i', s)
        s = re.sub('[ÍÌỈĨỊ]', 'I', s)
        s = re.sub('[úùủũụưứừửữự]', 'u', s)
        s = re.sub('[ÚÙỦŨỤƯỨỪỬỮỰ]', 'U', s)
        s = re.sub('[ýỳỷỹỵ]', 'y', s)
        s = re.sub('[ÝỲỶỸỴ]', 'Y', s)
        s = re.sub('đ', 'd', s)
        s = re.sub('Đ', 'D', s)
        return s
    def form_dataset(self, num_img_train_per_person : int):
        """
        :param num_img_train_per_person: number of image/person need for training
        :return: make a dir dataset with Structure:
        <Train>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
        <Test>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
        """

        dataset_dir = os.path.join(self.root, 'dataset')
        os.remove(dataset_dir) # remove to avoid conflict

        self.mkdir(dataset_dir)
        for person_name in os.listdir(self.path):
            # create train and test dir in dataset dir
            train_dir = os.path.join(dataset_dir, person_name, 'Train')
            test_dir = os.path.join(dataset_dir, person_name, 'Test')
            self.mkdir(train_dir)
            self.mkdir(test_dir)

            origin_dir = os.path.join(self.path, person_name)
            img_names = os.listdir(origin_dir)
            np.random.shuffle(img_names)
            print(img_names)
            # Train
            for i, img_name in enumerate(img_names[:num_img_train_per_person]):
                os.system('copy {} {}'.format(os.path.join(origin_dir, img_name),
                                              os.path.join(train_dir, 'img_{}.{}'.format(i, img_name[-3:]))))
            # Test
            for i, img_name in enumerate(img_names[num_img_train_per_person:]):
                os.system('copy {} {}'.format(os.path.join(origin_dir, img_name),
                                              os.path.join(test_dir, 'img_{}.{}'.format(i, img_name[-3:]))))
    def mkdir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
if __name__ == '__main__':
    crawler = CrawlInsta()
    crawler.form_dataset(2)
    arr_acc = ['sontungmtp', 'chipupu', 'tuilatranthanhday', 'huoggiangggg', 'ninh.duong.lan.ngoc',\
               'wonhari', 'kinglnd', 'dieu_nhiii', 'chaubui_', 'bichphuongofficial', 'hongocha', 'misthyyyy',\
               'quynhanhshyn_', 'sithanh', 'soobin.hoangson', 'ngocthao_official', 'baoanh0309', 'sam.ng_official', \
               'nguyenducphuc_', 'may__lily', 'imkhangan', 'minminmin0712', 'phodacbiet', 'loungu', 'hienho2620', \
               'ameliezilber', 'hoangyenchibi812', 'junpham', 'erikthanh_', 'angela.phuongtrinh', 'dienvien_duykhanh',\
               'kaitynguyen', 'den.vau', 'crisdevilgamer', 'manttien', 'gintuankiet', 'duymanh2909', 'diemmyvu',\
               'mienguyen', 'hoangthuylinhofficial', 'tu_hhao', 'maiquynhanh07', 'kienhoang254', 'lebong95', \
               'doanvanhau_1904', 'heominhon', 'tsun.sg', 'hhennie.official', 'afroswaggatrends_', 'phamquynhanh',\
               'harrylu92', 'chanchan.0411', 'kieutrinhxiu', 'phi.phuonganh', 'tronieeeeeeeee', 'hoaiann_', 'sunht',\
               'b.baothanh', 'btranofficial', 'quynhanh23', 'domylinh1310', 'tran3duy', 'trinh.phamm', 'hoangku', \
              'ngoctrinh89']
    # crawler.load_image('taylorswift', update = True)
    crawler.load_images(arr_acc, update=True)
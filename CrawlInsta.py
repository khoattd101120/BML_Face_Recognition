import os
import glob
import dlib
import cv2

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
        if update:
            os.system('instaloader --fast-update --no-video-thumbnails --no-videos --no-captions \
            --no-metadata-json --no-compress-json {}'.format(name))
        else:
            os.system('instaloader --no-video-thumbnails --no-videos --no-captions \
            --no-metadata-json --no-compress-json {} '.format(name))
        self.remove_redundant(name)
    def load_images(self, names, update = True):
        for name in names:
            self.load_image(name, update)
    def remove_redundant(self, folder):
        for redundant in glob.glob(os.path.join(self.path, folder, '*.*')):
            if not (redundant.endswith('.jpg') or redundant.endswith('.png')):
                os.remove(redundant)
            else: # remove image not contains face
                img = cv2.imread(redundant)
                small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]
                faces = self.face_detector(rgb_small_frame, 1)
                if len(faces) == 0:
                    os.remove(redundant)
        try:
            os.remove(os.path.join(self.path, folder, 'id'))
        except:
            pass

if __name__ == '__main__':
    crawler = CrawlInsta()
    crawler.load_image('taylorswift', update = True)
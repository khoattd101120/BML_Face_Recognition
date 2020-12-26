import cv2
import face_recognition
import numpy as np
import time
class FeatureExtraction:
    def __init__(self, img_path: str):
        self.img_path = img_path
        self.img = cv2.imread(img_path)

    def face_locations(self, max_num_face=1):
        """
        :param max_num_face (None,or 1; default: 1): 1 if find 1 face, None if get all faces
        :return: make a dir dataset with Structure:
        """
        image = face_recognition.load_image_file(self.img_path)
        faces = face_recognition.face_locations(self.img)
        if max_num_face:
            return faces[0]
        return faces

    def hog(self):
        location = self.face_locations()
        top, right, bottom, left = location
        face = fe.img[top: bottom, left:right, :]
        # face = fe.img[right: right + left, top:top + bottom, :]
        face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)/255.0

        kernel_x = np.array([-1, 0, 1])
        kernel_y = np.array([-1, 0, 1]).T

        kernel_x = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]])
        face = cv2.resize(face, (154, 154))

        gx = cv2.filter2D(face, -1, kernel_x)
        gy = cv2.filter2D(face, -1, kernel_y)

        # ang = np.arctan(gy/gx)
        # mag = ((gx**2 + gy**2)**.5)

        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        print(mag.shape)

        cv2.imshow('image', mag)
        cv2.waitKey(100000000)
    def HOG(self):
        # importing required libraries
        from skimage.io import imread
        from skimage.transform import resize
        from skimage.feature import hog
        from skimage import exposure
        import matplotlib.pyplot as plt

        location = self.face_locations()
        top, right, bottom, left = location
        face = fe.img[top: bottom, left:right, :]

        # resizing image
        resized_img = resize(face, (128 * 4, 64 * 4))
        # plt.axis("off")
        # plt.imshow(resized_img)
        # print(resized_img.shape)
        # creating hog features
        fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True, multichannel=True)
        plt.axis("off")
        plt.imshow(hog_image, cmap="gray")
        plt.show()
if __name__ == '__main__':
    s = time.time()
    fe = FeatureExtraction(
        r'C:\Workplaces\BML\Project\BML_Face_Recognition\dataset\hoangthuylinhofficial\Train\img_1.jpg')
    fe.HOG()
    print(time.time() - s)

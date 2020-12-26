import os
import cv2
import shutil
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

# Simple and common augmentation sequence
ia.seed(1)


def augment_images(image, batch_size=8, is_face=True):
    """
    Custom by: Tin Nguyen VinBIGDATA
    The array has shape (batch_size, image_size, image_size, 3) and dtype uint8.
    :param image:
    :param batch_size:
    :param is_face:
    :return: Images that processed by augmentation method
    """
    # Check image shape
    if is_face:
        if image.shape[0] != 112 or image.shape[1] != 112:
            image = cv2.resize(image, (112, 112))
    images = np.array([np.copy(image) for _ in range(batch_size)], dtype=np.uint8)

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.15)),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.LinearContrast((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True)  # apply augmenters in random order

    images_aug = seq(images=images)
    return images_aug


def process_augmentation_images(root_input_path, number_images_in_each_folder, batch_size, is_face=True):
    """

    :param is_face:
    :param root_input_path:
    :param number_images_in_each_folder:
    :param batch_size:
    :return:
    """

    output = os.path.join(os.path.dirname(root_input_path), 'processed_aug')
    if not os.path.exists(output):
        os.mkdir(output)
    for folder_name in os.listdir(root_input_path):
        path_folder_input = os.path.join(root_input_path, folder_name)
        path_folder_output = os.path.join(output, folder_name)
        if not os.path.exists(path_folder_output):
            os.mkdir(path_folder_output)

        # Augment images
        for img_name in os.listdir(path_folder_input):
            path_img = os.path.join(path_folder_input, img_name)
            img = cv2.imread(path_img)
            lst_img_aug = augment_images(img, batch_size=batch_size, is_face=is_face)
            for idx, aug_img in enumerate(lst_img_aug):
                cv2.imwrite(os.path.join(path_folder_output, img_name.split('.')[0] + str(idx) + '.jpg'), aug_img)
        # Check the number images in output dir, if it is greater than NUMBER_IMAGES_IN_EACH_FOLDER, delete few images!
        N = NUMBER_IMAGES_IN_EACH_FOLDER - len(os.listdir(path_folder_input))
        while len(os.listdir(path_folder_output)) > N:
            name_del = os.listdir(path_folder_output)[0]
            path_img_del = os.path.join(path_folder_output, name_del)
            os.remove(path_img_del)
        # Copy original images to output dir
        for img_name in os.listdir(path_folder_input):
            path_img = os.path.join(path_folder_input, img_name)
            shutil.copy(path_img, os.path.join(path_folder_output, img_name))


if __name__ == '__main__':
    ROOT_INPUT_PATH = r'C:\Users\tinnvt\Documents\BasicML\Project\Face ' \
                      r'Recognition\BML_Face_Recognition\face_recognition_v3\FaceRecog\Dataset\FaceData\raw '
    NUMBER_IMAGES_IN_EACH_FOLDER = 60
    BATCH_SIZE = 9
    IMAGE_SIZE = 112
    process_augmentation_images(ROOT_INPUT_PATH, NUMBER_IMAGES_IN_EACH_FOLDER, BATCH_SIZE, is_face=True)

from __future__ import print_function
import shutil
import os
import face_recognition.api as face_recognition

if __name__ == "__main__":
    path_folder = r'C:\Users\tinnvt\Documents\BasicML\Project\Face Recognition\BML_Face_Recognition\images'
    path_remove_images = r'C:\Users\tinnvt\Documents\BasicML\Project\Face Recognition\BML_Face_Recognition\removed_images'
    for sub_folder_name in os.listdir(path_folder):
        sub_folder_path = os.path.join(path_folder, sub_folder_name)
        sub_folder_remove = os.path.join(path_remove_images, sub_folder_name)
        if not os.path.exists(sub_folder_remove):
            os.mkdir(sub_folder_remove)

        for img_name in os.listdir(sub_folder_path):
            try:
                path_img = os.path.join(sub_folder_path, img_name)

                unknown_image = face_recognition.load_image_file(path_img)
                face_locations = face_recognition.face_locations(unknown_image, number_of_times_to_upsample=1, model="hog")
                if len(face_locations) == 0:
                    print('No faces in this image!', path_img)
                    shutil.move(path_img, os.path.join(sub_folder_remove, img_name))
            except Exception as e:
                print(f'Image {img_name} has error {e}')

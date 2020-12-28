# BML_Face_Recognition
## Team Members
* Nguyen Van Trung Tin
* Le Binh
* Nguyen The Duy
* Nguyen Thi Tuyet Hanh


## Dataset:  
* You must download dataset in this link
```
# onedrive.com
https://ued-my.sharepoint.com/:f:/g/personal/311011141145_ued_vn/EjUi7PLRBY5JjPFn-kKE5uIBBGJ5yZTJEPG3l1cYoYRdSw?e=NEmyHS
```

## Our best version: face_recognition_v4
* We created file .pikle to save embedding features of images. If you have new datasets, you can add it in folder "face_recognition\dataset", then run command lines to create new file embedding.
```
# python
cd face_recognition_v4\src
python FaceEmbedding.py
```  


## Feel excited to experience my UI demo
```
# python
cd face_recognition_v4\gui
python app-gui.py
```

## Here is the UI flow
# Add a User
Choose "Add a User" if you want to add a new member. Enter name and click next. We will capture 10 image in 5 seconds if you click "Capture dataset" and store these image in BML_Face_Recognition/face_recognition_v4/dataset_tmp/<name>

And then, click "Train the model" to train and update your model. In this step, we perform "Face Dectection" -> "Face Embedding", store/update embedded image and its label in  BML_Face_Recognition/face_recognition_v4/src/outputs/embeddings.pickle. The images in dataset_tmp is also moved to dataset.

New model will be train using these embedded image and store in  BML_Face_Recognition/face_recognition_v4/src/outputs/<model_type>_model.pickle

# Check a user
In this feature, the lastest model will be loaded to perform face recognition. The embedding step work same as in "Add a user"


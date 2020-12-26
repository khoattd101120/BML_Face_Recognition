# FaceRecog
Nhận diện khuôn mặt khá chuẩn xác bằng MTCNN và Facenet!

Article link: http://ainoodle.tech/2019/09/11/face-recog-2-0-nhan-dien-khuon-mat-trong-video-bang-mtcnn-va-facenet/

### Preprocess cắt khuôn mặt từ ảnh gốc
```
python sources\align_dataset_mtcnn.py  Dataset\FaceData\raw Dataset\FaceData\processed --image_size 112 --margin 32  --random_order --gpu_memory_fraction 0.25
```

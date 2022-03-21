import random
import os
import cv2
import time








def start_capture(name, recognizer):
    s = time.time()
    cap = cv2.VideoCapture(0)
    print('time:', time.time() - s)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # 450 #cv2.CAP_PROP_FRAME_WIDTH
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 580 #cv2.CAP_PROP_FRAME_HEIGHT
    a = random.randint(30, height - 100)
    b = random.randint(0, width - 100)

    print('time:', time.time() - s)

    cnt = 0
    capture = 0
    num_img = 0
    data_dir = os.path.join(os.getcwd(), '../dataset_tmp')
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    name = name.lower().replace(' ', '_')
    data_dir = os.path.join(os.getcwd(), '../dataset_tmp', name)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    while True:
        key = cv2.waitKey(int(1000 // frame_rate))
        cnt += 1

        ret, frame = cap.read()
        defaut_image = frame

        if cnt % int(frame_rate * 0.5) == 0:  # capture image every 0.5 second
            a = random.randint(30, width - 200)
            b = random.randint(30, height - 200)
            if capture:
                ret, frame = cap.read()
                input = recognizer.face_model.get_input(frame)

                if input is None:
                    cnt = num_img * int(frame_rate * 0.5)
                    cv2.imshow('image', frame)
                    continue
                bbox, face = input

                x, y, w, h = map(int, bbox)

                font = cv2.FONT_HERSHEY_PLAIN
                frame = cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

                defaut_image = defaut_image[y: h, x: w]
                num_img = cnt // int(frame_rate * 0.5)
                cv2.imwrite(os.path.join(data_dir, f'img_{num_img}.jpg'), defaut_image)

                if num_img == 10:
                    break

        if not capture:
            cv2.putText(frame, "Press Space to start", (a, b), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        if key & 0xFF == ord(' '):  # space
            capture = 1
            cnt = 0
        if key & 0xFF == ord('q'):  # space
            break

        cv2.imshow('image', frame)
    cv2.destroyAllWindows()
    cap.release()
    print("Done!")
    return num_img


if __name__ == '__main__':
    start_capture()

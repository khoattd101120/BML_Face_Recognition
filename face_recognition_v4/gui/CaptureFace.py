import random

def start_capture(name, recognizer):
    import cv2
    cap = cv2.VideoCapture(0)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = 450
    height = 580
    a = random.randint(30, height - 100)
    b = random.randint(0, width - 100)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cnt = 0
    capture = 0

    while True:
        key = cv2.waitKey(int(1000 // frame_rate))
        cnt += 1

        ret, frame = cap.read()
        defaut_image = frame

        if cnt == int(frame_rate * 0.5): # capture image every 0.5 second
            cnt = 0
            a = random.randint(30, width - 200)
            b = random.randint(30, height - 200)
            if capture:
                ret, frame = cap.read()
                input = recognizer.face_model.get_input(frame)

                if input is None:
                    return input
                bbox, face = input

                x, y, w, h = map(int, bbox)

                font = cv2.FONT_HERSHEY_PLAIN
                frame = cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)



        if not capture:
            cv2.putText(frame, "Press Space to start", (a, b), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        if key & 0xFF == ord(' '):  # space
            capture = 1
        if key & 0xFF == ord('q'):  # space
            cv2.destroyAllWindows()
            break

        cv2.imshow('image', frame)
    cap.release()
    print("Done!")
    return

    # path = "./data/" + name
    #
    # num_of_images = 0
    # detector = cv2.CascadeClassifier("./data/haarcascade_frontalface_default.xml")
    # try:
    #     os.makedirs(path)
    # except:
    #     print('Directory Already Created')
    # vid = cv2.VideoCapture(0)
    # while True:
    #
    #     ret, img = vid.read()
    #     new_img = None
    #     grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     face = detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
    #     for x, y, w, h in face:
    #         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
    #         cv2.putText(img, "Face Detected", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
    #         cv2.putText(img, str(str(num_of_images) + " images captured"), (x, y + h + 20),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
    #         new_img = img[y:y + h, x:x + w]
    #     cv2.imshow("FaceDetection", img)
    #     key = cv2.waitKey(1) & 0xFF
    #
    #     try:
    #         cv2.imwrite(str(path + "/" + str(num_of_images) + name + ".jpg"), new_img)
    #         num_of_images += 1
    #     except:
    #
    #         pass
    #     if key == ord("q") or key == 27 or num_of_images > 310:
    #         break
    # cv2.destroyAllWindows()
    # return num_of_images


if __name__ == '__main__':
    start_capture('hâh')
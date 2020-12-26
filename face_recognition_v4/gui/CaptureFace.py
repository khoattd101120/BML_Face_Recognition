import random
import cv2


def start_capture(name, recognizer):
    cap = cv2.VideoCapture(0)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    path = "./data/" + name
    num_of_images = 0
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
        new_img = None

        if cnt == int(frame_rate * 0.5):  # capture image every 0.5 second
            cnt = 0
            a = random.randint(30, width - 200)
            b = random.randint(30, height - 200)
            if capture:
                ret, frame = cap.read()
                input = recognizer.face_model.get_input(frame)

                if input is None:
                    return input
                bbox, face = input
                print(face.shape)

                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

                try:
                    cv2.imwrite(str(path + "/" + str(num_of_images) + name + ".jpg"), new_img)
                    num_of_images += 1
                except Exception as error:
                    print(error)

        if not capture:
            cv2.putText(frame, "Press Space to start", (a, b), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0))
        if key & 0xFF == ord(' '):  # space
            capture = 1

        if key & 0xFF == ord('q'):  # space
            cv2.destroyAllWindows()
            break

        cv2.imshow('image', frame)
    cap.release()
    print("Processed successfully!")
    return


if __name__ == '__main__':
    start_capture()

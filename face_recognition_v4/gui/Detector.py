import cv2
import time
import numpy as np

def main_app(name, recognizer):
        ss = 0
        cntt = 0
        # recognizer =CreateClassifier()
        cap = cv2.VideoCapture(0)
        pred = 0
        while True:
            ret, frame = cap.read()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            s = time.time()
            embed_out = recognizer.face_embedding.embed_face(frame)

            if not embed_out is None:

                bbox, embedding = embed_out

                x, y, w, h = map(int, bbox)

                model = recognizer.model

                proba = model.predict_proba([embedding])[0]
                id = np.argmax(proba)
                print(proba.round(3))
                print(recognizer.labels_unique)
                name = recognizer.labels_unique[id]
                confidence = proba[id]

                cntt += 1
                ss += time.time() - s
                # print(__name__, confidence,
                #       1.05/len(recognizer.labels_unique))
                if confidence > 1.05/len(recognizer.labels_unique):
                    text = name.upper()
                    font = cv2.FONT_HERSHEY_PLAIN
                    frame = cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                    frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

                else:
                    text = "UnknownFace"
                    font = cv2.FONT_HERSHEY_PLAIN
                    frame = cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
                    frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 0,255), 1, cv2.LINE_AA)
            if cntt == 500:
                break
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

            cv2.imshow("image", frame)
        cap.release()
        cv2.destroyAllWindows()
        print(__name__, 'mean time:', ss/cntt)


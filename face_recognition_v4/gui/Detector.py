import cv2
from time import sleep
from PIL import Image 
import numpy as np

def main_app(name, recognizer):

        # recognizer =CreateClassifier()
        cap = cv2.VideoCapture(0)
        pred = 0
        while True:
            ret, frame = cap.read()
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            embed_out = recognizer.face_embedding.embed_face(frame)

            if not embed_out is None:

                bbox, embedding = embed_out

                x, y, w, h = map(int, bbox)

                model = recognizer.face_embedding.load_pickle(r'..\src\outputs\svm_model.pickle')

                proba = model.predict_proba([embedding])[0]
                id = np.argmax(proba)

                name = recognizer.labels_unique[id]
                confidence = proba[id]

                if confidence > 0.1:
                    text = name.upper()
                    font = cv2.FONT_HERSHEY_PLAIN
                    frame = cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                    frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

                else:
                    text = "UnknownFace"
                    font = cv2.FONT_HERSHEY_PLAIN
                    frame = cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
                    frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 0,255), 1, cv2.LINE_AA)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

            cv2.imshow("image", frame)
        cap.release()
        cv2.destroyAllWindows()


import sys
sys.path.append('C:\Workplaces\BML\Project\BML-Face-Recognition-with-InsightFace\src')
sys.path.append('C:\Workplaces\BML\Project\BML-Face-Recognition-with-InsightFace\insightface\deploy')
sys.path.append('C:\Workplaces\BML\Project\BML-Face-Recognition-with-InsightFace\insightface\common')
from CreateClassifier import CreateClassifier
import cv2
from time import sleep
from PIL import Image 
import numpy as np

def main_app(name):

        recognizer =CreateClassifier()
        cap = cv2.VideoCapture(0)
        pred = 0
        while True:
            ret, frame = cap.read()
            default_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bbox, embedding = recognizer.face_embedding.embed_face(default_img)

            x, y, w, h = map(int, bbox)
            model = recognizer.face_embedding.load_pickle(r'C:\Workplaces\BML\Project\BML-Face-Recognition-with-InsightFace\src\outputs\svm_model.pickle')

            proba = model.predict_proba([embedding])[0]
            id = np.argmax(proba)
            print(recognizer.labels_unique)
            name = recognizer.labels_unique[id]
            confidence = proba[id]
            pred = 0
            if True:
                #if u want to print confidence level
                        #confidence = 100 - int(confidence)
                        pred += +1
                        print('sadf----------', name)
                        text = name.upper()
                        font = cv2.FONT_HERSHEY_PLAIN
                        frame = cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                        frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

            else:
                        pred += -1
                        text = "UnknownFace"
                        font = cv2.FONT_HERSHEY_PLAIN
                        frame = cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
                        frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 0,255), 1, cv2.LINE_AA)

            cv2.imshow("image", frame)


            if cv2.waitKey(20) & 0xFF == ord('q'):
                print(pred)
                if pred > 0 :
                    dim =(124,124)
                    img = cv2.imread(f".\\data\\{name}\\{pred}{name}.jpg", cv2.IMREAD_UNCHANGED)
                    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                    cv2.imwrite(f".\\data\\{name}\\50{name}.jpg", resized)
                    Image1 = Image.open(f".\\2.png")

                    # make a copy the image so that the
                    # original image does not get affected
                    Image1copy = Image1.copy()
                    Image2 = Image.open(f".\\data\\{name}\\50{name}.jpg")
                    Image2copy = Image2.copy()

                    # paste image giving dimensions
                    Image1copy.paste(Image2copy, (195, 114))

                    # save the image
                    Image1copy.save("end.png")
                    frame = cv2.imread("end.png", 1)

                    cv2.imshow("Result",frame)
                    cv2.waitKey(5000)
                break


        cap.release()
        cv2.destroyAllWindows()




from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import time
import numpy as np




def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)


print("[INFO] loading face detector model...")
prototxtPath = 'face_detector/deploy.prototxt'
weightsPath = 'face_detector/res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model('mask_detector.model')

def main_app(name, recognizer):
    ss = 0
    cntt = 0
    # recognizer =CreateClassifier()
    cap = cv2.VideoCapture(0)
    # pred = 0
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
            # print(proba.round(3))
            # print(recognizer.labels_unique)
            name = recognizer.labels_unique[id]
            print(name)
            confidence = proba[id]

            cntt += 1
            ss += time.time() - s
            # print(__name__, confidence,
            #       1.05/len(recognizer.labels_unique))
            print(confidence)
            print(1.05 / len(recognizer.labels_unique))

            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
            for (box, pred) in zip(locs, preds):
                (mask, withoutMask) = pred
                label = "Mask" if mask > withoutMask else "No Mask"
                if mask < withoutMask:

                    # if confidence > 1.05 / len(recognizer.labels_unique):
                    if confidence > 0.8:
                        text = name.upper()
                        font = cv2.FONT_HERSHEY_PLAIN
                        frame = cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
                        frame = cv2.putText(frame, text, (x, y - 4), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

                    else:
                        text = "UnknownFace"
                        font = cv2.FONT_HERSHEY_PLAIN
                        frame = cv2.rectangle(frame, (x, y), (w, h), (0, 0, 255), 2)
                        frame = cv2.putText(frame, text, (x, y - 4), font, 1, (0, 0, 255), 1, cv2.LINE_AA)

                else:
                    
                    # text = name.upper()
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                    font = cv2.FONT_HERSHEY_PLAIN
                    frame = cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
                    frame = cv2.putText(frame, label, (x, y - 4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
        if cntt == 500:
            break
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

        cv2.imshow("image", frame)
    cap.release()
    cv2.destroyAllWindows()
    print(__name__, 'mean time:', ss / cntt)

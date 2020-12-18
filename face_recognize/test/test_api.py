# import requests
# import json
#
# base_link = "http://34.227.31.49:3000/customer-coming"
#
# headers = {'content-type': 'application/json'}
#
# data = {"id": None,
#                         "gender": "male",
#                         "emotion": "happy"}
#
#
# response = requests.post(base_link, data=json.dumps(data), headers=headers)
#
# print(response.text)


import cv2


video_capture = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    winname = "Test"
    cv2.namedWindow(winname)        # Create a named window
    cv2.moveWindow(winname, 1250,500)  # Move it to (40,30)
    cv2.imshow(winname, frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()




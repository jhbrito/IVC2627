import time
import cv2
import numpy as np
import os

classifier_folder = cv2.data.haarcascades
classifier_file = 'haarcascade_frontalface_alt.xml'
face_detector = cv2.CascadeClassifier(os.path.join(classifier_folder, classifier_file))

cap = cv2.VideoCapture()
before = 0

while True:
    now = time.time()
    fps = 1 / (now - before)
    before = now

    if not cap.isOpened():
        cap.open(0)

    ret, frame = cap.read()

    frame_mirror = frame[:,::-1,:]
    faces = face_detector.detectMultiScale(frame_mirror, scaleFactor=1.1)

    image_faces = frame_mirror.copy()

    for face in faces:
        x, y, w, h = face

        cv2.rectangle(image_faces,
                      (x, y),
                      (x+w, y+h),
                      (255, 0, 0),
                      2)
    text_to_show = "{} fps".format(np.round(fps))
    cv2.putText(image_faces,
                text_to_show,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
                )
    cv2.imshow('faces', image_faces)

    c = cv2.waitKey(1)
    if c == 27 or c == 'q':
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()

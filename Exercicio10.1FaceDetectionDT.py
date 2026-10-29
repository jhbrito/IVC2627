import time
import cv2
import numpy as np
import os

use_cam = False

video_folder = "Files"
video_file = "vtest.avi"

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor.getDefaultPeopleDetector())

if use_cam:
    cap = cv2.VideoCapture()
else:
    cap = cv2.VideoCapture(os.path.join(video_folder, video_file))

before = 0

while True:
    now = time.time()
    fps = 1 / (now - before)
    before = now

    if not cap.isOpened():
        cap.open(0)

    ret, frame = cap.read()

    frame_mirror = frame[:,::-1,:]
    persons, _ = hog.detectMultiScale(frame_mirror,
                                      scale=1.1,
                                      winStride=(8,8))

    image_faces = frame_mirror.copy()

    for person in persons:
        x, y, w, h = person

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

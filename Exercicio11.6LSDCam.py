import time
import cv2 as cv2
import os
import numpy as np

use_cam = True
folder ="Files"
file = "vtest.avi"

if use_cam:
    cap = cv2.VideoCapture()
else:
    cap = cv2.VideoCapture(os.path.join(folder, file))

lsd = cv2.createLineSegmentDetector()

before = 0

while True:
    now = time.time()
    fps = 1/ (now - before)
    before = now

    if not(cap.isOpened()):
        cap.open(0)

    ret, frame = cap.read()
    if ret:
        h, w, c = frame.shape

        if use_cam:
            frame_mirror = frame[:, ::-1, :]
        else:
            frame_mirror = frame

        gray_image = cv2.cvtColor(frame_mirror, cv2.COLOR_BGR2GRAY)

        #Detect lines in the image
        lines = lsd.detect(gray_image)[0] #Position 0 of the returned tuple are the detected lines

        image_with_lines = frame_mirror.copy()

        #Draw detected lines in the image
        # drawn_img = lsd.drawSegments(image_with_lines,lines)
        for line in lines:
            line = line[0]
            p1 = np.array((line[0], line[1]))
            p2 = np.array((line[2], line[3]))
            l = np.sqrt(np.sum((p1 - p2)**2))
            if l>25:
                cv2.line(image_with_lines,
                         (int(line[0]), int(line[1])),
                         (int(line[2]), int(line[3])),
                         (0, 255, 0),
                         1,
                         cv2.LINE_AA)

        text_to_show = "{} fps".format(np.round(fps))
        cv2.putText(img=image_with_lines,
                    text=text_to_show,
                    org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 255, 0),
                    thickness=2)

        cv2.imshow("Lines", image_with_lines)

        c = cv2.waitKey(1)
        if c == 27:
            break
    else:
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()

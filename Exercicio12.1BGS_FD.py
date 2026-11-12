import os
import cv2
import numpy as np
import time

use_cam = False
folder ="Files"
file = "vtest.avi"

if use_cam:
    cap = cv2.VideoCapture()
else:
    cap = cv2.VideoCapture(os.path.join(folder, file))

if not(cap.isOpened()):
    cap.open(0)
ret, frame = cap.read()

if use_cam:
    frame_mirror = frame[:, ::-1, :]
else:
    frame_mirror = frame

previous_frame = cv2.cvtColor(frame_mirror, cv2.COLOR_BGR2GRAY)
threshold = 10
before = 0

def on_trackbar_threshold(val):
    global threshold
    threshold = val


cv2.namedWindow("Image")
cv2.createTrackbar("Threshold",
                   "Image",
                   threshold,
                   255,
                   on_trackbar_threshold)

while True:
    now = time.time()

    if now != before:
        fps = 1/ (now - before)
    else:
        fps = np.inf
    before = now

    ret, frame = cap.read()
    if ret:
        h, w, c = frame.shape

        if use_cam:
            frame_mirror = frame[:, ::-1, :]
        else:
            frame_mirror = frame

        gray_image = cv2.cvtColor(frame_mirror, cv2.COLOR_BGR2GRAY)

        dif = np.abs(gray_image - previous_frame)
        _, fg_mask = cv2.threshold(src=dif,
                                   thresh=threshold,
                                   maxval=1,
                                   type=cv2.THRESH_BINARY)
        segmented_image = frame_mirror.copy()
        segmented_image[:, :, 0] = segmented_image[:, :, 0] * fg_mask
        segmented_image[:, :, 1] = segmented_image[:, :, 1] * fg_mask
        segmented_image[:, :, 2] = segmented_image[:, :, 2] * fg_mask

        text_to_show = str(int(np.round(fps))) + " fps"
        cv2.putText(img=segmented_image,
                    text=text_to_show,
                    org=(5, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    thickness=1)

        cv2.imshow("Image", segmented_image)
        cv2.imshow("Dif", dif)
        cv2.imshow("Foreground Mask", fg_mask*255)

        previous_frame = gray_image

        if use_cam:
            c = cv2.waitKey(1)
        else:
            c = cv2.waitKey(33)

        if c == 27:
            break
    else:
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()

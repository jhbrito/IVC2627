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

background_mog = cv2.createBackgroundSubtractorMOG2()
N = background_mog.getNMixtures()
print(N)
background_mog.setNMixtures(7)

background_knn = cv2.createBackgroundSubtractorKNN()


before = 0

cv2.namedWindow("Image")


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

        fg_mask_mog = background_mog.apply(frame_mirror)
        fg_mask_knn = background_knn.apply(frame_mirror)

        _, fg_mask_mog_t = cv2.threshold(fg_mask_mog, 128, 1, cv2.THRESH_BINARY)
        _, fg_mask_knn_t = cv2.threshold(fg_mask_knn, 128, 1, cv2.THRESH_BINARY)

        background_subtraction_mog = frame_mirror.copy()
        background_subtraction_mog[:, :, 0] = background_subtraction_mog[:, :, 0] * fg_mask_mog_t
        background_subtraction_mog[:, :, 1] = background_subtraction_mog[:, :, 1] * fg_mask_mog_t
        background_subtraction_mog[:, :, 2] = background_subtraction_mog[:, :, 2] * fg_mask_mog_t

        background_subtraction_knn = frame_mirror.copy()
        background_subtraction_knn[:, :, 0] = background_subtraction_knn[:, :, 0] * fg_mask_knn_t
        background_subtraction_knn[:, :, 1] = background_subtraction_knn[:, :, 1] * fg_mask_knn_t
        background_subtraction_knn[:, :, 2] = background_subtraction_knn[:, :, 2] * fg_mask_knn_t

        frame_mirror_show = frame_mirror.copy()
        text_to_show = str(int(np.round(fps))) + " fps"
        cv2.putText(img=frame_mirror_show,
                    text=text_to_show,
                    org=(5, 15),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    thickness=1)
        cv2.imshow("Image", frame_mirror_show)
        cv2.imshow("MOG FG Mask", fg_mask_mog)
        cv2.imshow("KNN FG Mask", fg_mask_knn)
        cv2.imshow("MOG BGS", background_subtraction_mog)
        cv2.imshow("KNN BGS", background_subtraction_knn)

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

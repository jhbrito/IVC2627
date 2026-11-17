import cv2
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input",
                    type=str,
                    default="vtest.avi",
                    help="Missing: --input: video file")
parser.add_argument("--folder",
                    type=str,
                    default="Files",
                    help="Missing: --folder: folder where files are stored")

args = parser.parse_args()

cap = cv2.VideoCapture(os.path.join(args.folder, args.input))

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

lucas_kanade_params = dict(winSize=(15, 15),
                           maxLevel=2,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (100, 3))

ret, old_frame = cap.read()
old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_frame_gray, mask=None, **feature_params)

mask = np.zeros_like(old_frame)

while True:
    ret, new_frame = cap.read()
    if not ret:
        print("No more frames")
        break

    new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame_gray,
                                           new_frame_gray,
                                           p0,
                                           None,
                                           **lucas_kanade_params)
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask,
                        (int(a),int(b)),
                        (int(c),int(d)),
                        color[i].tolist(),
                        2)
        new_frame = cv2.circle(new_frame,
                               (int(a),int(b)),
                               5,
                               color[i].tolist(),
                               -1)
    img = cv2.add(new_frame, mask)
    cv2.imshow("Mask", mask)
    cv2.imshow("Frame", img)

    k = cv2.waitKey(30)
    if k == 27:
        break
    old_frame_gray = new_frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()



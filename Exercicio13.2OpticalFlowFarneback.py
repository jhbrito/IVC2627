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

ret, frame = cap.read()
previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No more frames")
        break

    next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev=previous_frame,
                                        next=next_frame,
                                        flow=None,
                                        pyr_scale=0.25,
                                        levels=1,
                                        winsize=5,
                                        iterations=1,
                                        poly_n=5,
                                        poly_sigma=1.2,
                                        flags=0)
    flow_norm = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)
    flow_norm_normalized = cv2.normalize(flow_norm,
                                         None,
                                         0.0,
                                         1.0,
                                         cv2.NORM_MINMAX)

    # cv2.imshow("flow_norm_normalized", flow_norm_normalized)

    frame_show = frame.copy()
    frame_show[:, :, 0] = frame[:, :, 0]/4 + frame[:, :, 0]*(3/4) * flow_norm_normalized
    frame_show[:, :, 1] = frame[:, :, 1]/4 + frame[:, :, 1]*(3/4) * flow_norm_normalized
    frame_show[:, :, 2] = frame[:, :, 2]/4 + frame[:, :, 2]*(3/4) * flow_norm_normalized

    cv2.imshow("Frame", frame_show)

    k = cv2.waitKey(30)
    if k == 27:
        break
    previous_frame = next_frame


cv2.destroyAllWindows()
cap.release()

import time
import cv2
import os
import numpy as np
from ultralytics import YOLO
import torch

use_cam = True
folder ="Files"
file = "vtest.avi"

if use_cam:
    cap = cv2.VideoCapture()
else:
    cap = cv2.VideoCapture(os.path.join(folder, file))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = YOLO("yolov8n.pt")
# model = YOLO("runs/detect/train2/weights/best.pt")
print("Known Classes ({})".format(len(model.names)))
for i in range(len(model.names)):
    print(model.names[i])
model.to(device)

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

        frame_mirrorRGB = cv2.cvtColor(frame_mirror, cv2.COLOR_RGB2BGR)
        objects = model.predict(frame_mirrorRGB, verbose=False)

        objects = objects[0]

        image_objects = frame_mirror.copy()

        for object in objects:
            (x1, y1, x2, y2, conf, class_id) = object.boxes.data[0]

            if conf > 0.5:
                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))
                if conf > 0.75:
                    if model.names[int(class_id)] == "person":
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                else:
                    color = (0, 0, 0)
                cv2.rectangle(img=image_objects,
                              pt1=p1,
                              pt2=p2,
                          color=color,
                          thickness=2)
                object_text = "{}:{:.2f}".format(model.names[int(class_id)], conf)
                cv2.putText(img=image_objects,
                            text=object_text,
                            org=(int(x1), int(y1)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=color,
                            thickness=2)


        text_to_show = "{} fps".format(np.round(fps))
        cv2.putText(img=image_objects,
                    text=text_to_show,
                    org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 255, 0),
                    thickness=2)


        cv2.imshow("objects", image_objects)
        c = cv2.waitKey(1)
        if c == 27:
            break
    else:
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()

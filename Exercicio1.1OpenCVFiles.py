import cv2
import os

Files = "Files"
file = "baboon.png"

file_path = os.path.join(Files, file)
image = cv2.imread(file_path)
image = image / 255.0
cv2.imshow("image", image)
cv2.waitKey(0)

video_file = "vtest.avi"
video_file_path = os.path.join(Files, video_file)
cap = cv2.VideoCapture(video_file_path)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        cv2.imshow("frame", frame)
        cv2.waitKey(100)
    else:
        break

if cap.isOpened():
    cap.release()

cv2.destroyAllWindows()

import cv2

cap = cv2.VideoCapture()

while True:
    if not cap.isOpened():
        cap.open(0)
    ret, frame = cap.read()

    h, w, c = frame.shape
    print(h,w)

    frame = frame[:, ::-1, :]
    frame_mirror = frame[:, ::-1, :]

    if ret:
        cv2.imshow("frame", frame)
        cv2.imshow("frame_mirror", frame_mirror)

        c = cv2.waitKey(1)
        if c == 27:
            break
    else:
        break

if cap.isOpened():
    cap.release()

cv2.destroyAllWindows()

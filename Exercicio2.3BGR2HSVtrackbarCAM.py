import cv2

h = 90

def on_trackbar_change_H(val):
    global h
    h = val


cv2.namedWindow("HSV")
cv2.createTrackbar("H",
                   "HSV",
                   h,
                   180,
                   on_trackbar_change_H)
cap = cv2.VideoCapture()

while True:
    if not cap.isOpened():
        cap.open(0)
    ret, frame = cap.read()


    frame_mirror = frame[:, ::-1, :]

    image_hsv = cv2.cvtColor(frame_mirror, cv2.COLOR_BGR2HSV)
    image_hsv[:, :, 0] = h
    image_hsv_bgr = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if ret:
        # cv2.imshow("frame", frame)
        cv2.imshow("HSV", image_hsv_bgr)

        c = cv2.waitKey(1)
        if c == 27:
            break
    else:
        break

if cap.isOpened():
    cap.release()

cv2.destroyAllWindows()

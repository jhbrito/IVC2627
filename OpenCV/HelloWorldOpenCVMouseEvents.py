import cv2

selection = None
selection_window = None
selection_start = None
track_window = None


def onmouse(event, x, y, flags, param):
    global selection, selection_window, selection_start, track_window

    if event == cv2.EVENT_RBUTTONDOWN:
        selection_start = None
        selection = None
        selection_window = None
        track_window = None
    if event == cv2.EVENT_LBUTTONDOWN:
        selection_start = (x, y)
        track_window = None
    if selection_start:
        xmin = min(x, selection_start[0])
        ymin = min(y, selection_start[1])
        xmax = max(x, selection_start[0])
        ymax = max(y, selection_start[1])
        selection = (xmin, ymin, xmax, ymax)
        selection_window = (xmin, ymin, xmax - xmin, ymax - ymin)
    if event == cv2.EVENT_LBUTTONUP:
        selection_start = None
        selection = None
        track_window = (xmin, ymin, xmax - xmin, ymax - ymin)


cap = cv2.VideoCapture()
if not(cap.isOpened()):
    cap.open(0)

_ret, frame = cap.read()


cv2.namedWindow('CAM')
cv2.setMouseCallback('CAM', onmouse)

while True:
    _ret, frame = cap.read()
    frame = frame[:, ::-1, :]
    vis = frame.copy()


    if selection:
        cv2.rectangle(vis, selection_window, (0, 255, 0), 2)

    if track_window and track_window[2] > 0 and track_window[3] > 0:
        cv2.rectangle(vis, track_window, (0, 0, 255), 2)

    cv2.imshow('CAM', vis)

    ch = cv2.waitKey(5)
    if ch == 27:
        break

cv2.destroyAllWindows()


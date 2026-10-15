import os
import cv2

folder = "Files"
file = "moedas.jpg"

image = cv2.imread(os.path.join(folder, file))
cv2.imshow("image", image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

threshhold1 = 100
threshhold2 = 200

def do_threshold():
    image_edges = cv2.Canny(image_gray, threshhold1, threshhold2)
    # image_edges = cv2.Canny(image_gray)
    cv2.imshow("Edges", image_edges)


def on_change_theshold1(val):
    global threshhold1
    threshhold1 = val
    do_threshold()

def on_change_theshold2(val):
    global threshhold2
    threshhold2 = val
    do_threshold()


cv2.namedWindow("Edges")
cv2.createTrackbar("Threshold1",
                   "Edges",
                   threshhold1,
                   255,
                   on_change_theshold1)
cv2.createTrackbar("Threshold2",
                   "Edges",
                   threshhold2,
                   255,
                   on_change_theshold2)
cv2.waitKey(0)
cv2.destroyAllWindows()

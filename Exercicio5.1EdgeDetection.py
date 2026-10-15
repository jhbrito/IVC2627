import os
import cv2
import numpy as np

folder = "Files"
file = "moedas.jpg"

image = cv2.imread(os.path.join(folder, file))

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = (image_gray / 255.0).astype(np.float32)

Mx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])
My = np.array([[-1, -2, -1],
               [ 0,  0,  0],
               [+1, +2, +1]])
dx = cv2.filter2D(src=image_gray, ddepth=-1, kernel=Mx)
dy = cv2.filter2D(src=image_gray, ddepth=-1, kernel=My)
gradient_magnitude = np.sqrt(dx ** 2 + dy ** 2)
gradiente_direction = np.arctan2(dy, dx) / np.pi

threshold = np.max(gradient_magnitude) / 2

def on_change_threshold(val):
    global threshold
    threshold = val/100.0
    ret, image_thresholded = cv2.threshold(src=gradient_magnitude,
                                           thresh=threshold,
                                           maxval=1.0,
                                           type=cv2.THRESH_BINARY)
    cv2.imshow("Edges", image_thresholded)


cv2.imshow("Image", image)
cv2.imshow("Dx", dx)
cv2.imshow("Dy", dy)
cv2.imshow("Gradient Magnitude", gradient_magnitude)

cv2.namedWindow("Edges")
cv2.createTrackbar("Threshold",
                   "Edges",
                   int(threshold*100),
                   int(np.max(gradient_magnitude)*100),
                   on_change_threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()

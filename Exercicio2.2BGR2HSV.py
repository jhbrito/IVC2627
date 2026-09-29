import cv2
import os

import numpy as np

Files = "Files"
file = "baboon.png"

file_path = os.path.join(Files, file)
image = cv2.imread(file_path)

image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
image_hsv[:,:,0] = 120
image_hsv_bgr = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

cv2.imshow("image", image)
cv2.imshow("image_hsv_bgr", image_hsv_bgr)

cv2.waitKey(0)
cv2.destroyAllWindows()

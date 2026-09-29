import cv2
import os

import numpy as np

Files = "Files"
file = "baboon.png"

file_path = os.path.join(Files, file)
image = cv2.imread(file_path)

def ivc_bgr_to_grayscale(src):
    R = src[:, :, 2]
    G = src[:, :, 1]
    B = src[:, :, 0]
    image_gray = R * 0.299 + G * 0.587 + B * 0.114
    image_gray = np.round(image_gray).astype(np.uint8)
    # image_gray = image_gray / 255.0
    return image_gray

image_gray = ivc_bgr_to_grayscale(image)

def ivc_gray_negative(src):
    if src.dtype == np.uint8:
        return 255 - src
    else:
        return 1.0 - src

def ivc_remove_red(src):
    dst = src.copy()
    dst[:, :, 2] = 0
    return dst


def ivc_remove_green(src):
    dst = src.copy()
    dst[:, :, 1] = 0
    return dst


def ivc_remove_blue(src):
    dst = src.copy()
    dst[:, :, 0] = 0
    return dst


image_gray_negative = ivc_gray_negative(image_gray)
image_no_red = ivc_remove_red(image)
image_no_green = ivc_remove_green(image)
image_no_blue = ivc_remove_blue(image)

cv2.imshow("image", image)
cv2.imshow("image_gray", image_gray)
cv2.imshow("image_gray_negative", image_gray_negative)
cv2.imshow("image_no_red", image_no_red)
cv2.imshow("image_no_green", image_no_green)
cv2.imshow("image_no_blue", image_no_blue)

cv2.waitKey(0)
cv2.destroyAllWindows()

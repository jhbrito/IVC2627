import cv2 as cv2
import os
import numpy as np

folder = "Files"
img = cv2.imread(os.path.join(folder, "building.jpg"))
cv2.imshow("Building", img)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Building Gray", gray_image)

image_with_lines = img.copy()

lsd = cv2.createLineSegmentDetector()

#Detect lines in the image
lines = lsd.detect(gray_image)[0] #Position 0 of the returned tuple are the detected lines

#Draw detected lines in the image
# drawn_img = lsd.drawSegments(image_with_lines,lines)
for line in lines:
    line = line[0]
    cv2.line(image_with_lines,
             (int(line[0]), int(line[1])),
             (int(line[2]), int(line[3])),
             (0, 255, 0),
             1,
             cv2.LINE_AA)

cv2.imshow("Lines", image_with_lines)

cv2.waitKey()

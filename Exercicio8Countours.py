import os
import cv2
import numpy as np

folder = "Files"
file = "moedas.jpg"

image = cv2.imread(os.path.join(folder, file))
cv2.imshow("image", image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray = image_gray / 255.0

cv2.imshow("image_gray", image_gray)

def on_trackbar_change(value):
    threshold = value / 100.0
    ret, image_thresholded = cv2.threshold(image_gray,
                                           threshold,
                                           1,
                                           cv2.THRESH_BINARY)
    image_thresholded = (image_thresholded * 255).astype("uint8")
    cv2.imshow("image_thresholded", image_thresholded)

    contours, hierarchy = cv2.findContours(image=image_thresholded,
                     mode=cv2.RETR_EXTERNAL,
                     method=cv2.CHAIN_APPROX_NONE)

    image_contours = np.zeros(image_gray.shape, dtype=np.uint8)
    cv2.drawContours(image=image_contours,
                     contours=contours,
                     contourIdx=-1,
                     color = 255,
                     thickness = -1)
    cv2.imshow("image_contours", image_contours)

    image_contours2 = np.zeros(image_gray.shape, dtype=np.uint8)
    image_contours_color = np.zeros(image.shape, dtype=np.uint8)

    for i in range(len(contours)):
        cv2.drawContours(image=image_contours2,
                         contours=contours,
                         contourIdx=i,
                         color = 255,
                         thickness = -1)
        cv2.drawContours(image=image_contours_color,
                         contours=contours,
                         contourIdx=i,
                         color = (0, 255, 0),
                         thickness=-1)
        contour = contours[i]

        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        print("contour {} area={} perimeter={}".format(i, area, perimeter))

        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(img=image_contours_color,
                   center=(cx, cy),
                   radius=3,
                   color=(255, 0, 0),
                   thickness=-1)

    max_area = 0
    max_index = 0

    for i in range(len(contours)):
        contour = contours[i]
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_index = i
    print("max_area={} max_index={}".format(max_area, max_index))
    cv2.drawContours(image=image_contours_color,
                     contours=contours,
                     contourIdx=max_index,
                     color = (0, 0, 255),
                     thickness = 2)




    cv2.imshow("image_contours2", image_contours2)
    cv2.imshow("image_contours_color", image_contours_color)













cv2.namedWindow("image_thresholded")
cv2.createTrackbar("Threshold",
                   "image_thresholded",
                   50,
                   100,
                   on_trackbar_change)
cv2.waitKey(0)
cv2.destroyAllWindows()

import os
import cv2
import numpy as np

folder = "Files"
file = "Sharbat_Gula.jpg"

image = cv2.imread(os.path.join(folder, file))
cv2.imshow("image", image)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("image_gray", image_gray)
harris_cornerness = cv2.cornerHarris(image_gray, 2, 3, 0.04)
harris_cornerness_image = cv2.normalize(harris_cornerness,
                                        None,
                                        0, 1.0,
                                        cv2.NORM_MINMAX)
cv2.imshow("harris_cornerness", harris_cornerness_image)

T = harris_cornerness.max() * 0.1

image_with_harris_corners = image.copy()
image_with_harris_corners[harris_cornerness > T] = [255, 0, 0]
cv2.imshow("image_with_harris_corners", image_with_harris_corners)

sift = cv2.SIFT_create()
sift_kp, sift_desc = sift.detectAndCompute(image, None)
image_sift = image.copy()
image_sift = cv2.drawKeypoints(image=image_sift,
                               keypoints=sift_kp,
                               outImage=None)
cv2.imshow("image_sift", image_sift)

# surf = cv2.xfeatures2d.SURF_create()
# surf_kp, surf_desc = surf.detectAndCompute(image, None)
# image_surf = image.copy()
# image_surf = cv2.drawKeypoints(image=image,
#                                keypoints=surf_kp,
#                                outImage=None)
# cv2.imshow("image_surf", image_surf)

orb = cv2.ORB_create()
orb_kp, orb_desc = orb.detectAndCompute(image, None)
image_orb = image.copy()
image_orb = cv2.drawKeypoints(image=image_orb, keypoints=orb_kp, outImage=None)
cv2.imshow("ORB", image_orb)

brisk = cv2.xfeatures2d.BRISK.create()
brisk_kp, brisk_desc = brisk.detectAndCompute(image, None)
image_brisk = image.copy()
image_brisk = cv2.drawKeypoints(image=image_brisk, keypoints=brisk_kp, outImage=None)
cv2.imshow("BRISK", image_brisk)

kaze = cv2.xfeatures2d.KAZE.create()
kaze_kp, kaze_desc = kaze.detectAndCompute(image, None)
image_kaze = image.copy()
image_kaze = cv2.drawKeypoints(image=image_kaze, keypoints=kaze_kp, outImage=None)
cv2.imshow("KAZE", image_kaze)


cv2.waitKey(0)
cv2.destroyAllWindows()

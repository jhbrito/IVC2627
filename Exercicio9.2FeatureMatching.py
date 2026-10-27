import cv2
import os

import numpy as np

resize_scale = 0.25

folder = "Files"
image1 = cv2.imread(os.path.join(folder, "boat1.jpg"))
image2 = cv2.imread(os.path.join(folder, "boat2.jpg"))

image1 = cv2.resize(image1, (int(image2.shape[1] * resize_scale), int(image2.shape[0] * resize_scale)))
image2 = cv2.resize(image2, (int(image2.shape[1] * resize_scale), int(image2.shape[0] * resize_scale)))

cv2.imshow("image1", image1)
cv2.imshow("image2", image2)

sift = cv2.SIFT_create()
sift_kp1, sift_desc1 = sift.detectAndCompute(image1, None)
sift_kp2, sift_desc2 = sift.detectAndCompute(image2, None)

image1_sift = image1.copy()
image1_sift = cv2.drawKeypoints(image=image1_sift, keypoints=sift_kp1, outImage=None)
image2_sift = image2.copy()
image2_sift = cv2.drawKeypoints(image=image2_sift, keypoints=sift_kp2, outImage=None)

cv2.imshow("SIFT1", image1_sift)
cv2.imshow("SIFT2", image2_sift)

h =  image1.shape[0]
w  = image1.shape[1] + image2.shape[1]
c = image1.shape[2]
image_dual = np.zeros( ( h, w, c ), dtype=np.uint8)
image_dual[:, 0:image1.shape[1], :] = image1
image_dual[:, image1.shape[1]: , :] = image2
# cv2.imshow("dual", image_dual)
#
# cv2.waitKey(1)

N1 = len(sift_kp1)
N2 = len(sift_kp2)

distances = np.zeros( ( N1, N2 ), dtype=np.float64 )

for i in range(N1):
    for j in range(N2):
        desc1 = sift_desc1[i]
        desc2 = sift_desc2[j]
        distance_L2 = np.sum((desc1 - desc2) ** 2)
        distances[i, j] = distance_L2

Th_L2 = 5000

for i in range(N1):
    if not (i % 100):
        print("{} / {}".format(i, N1))

    j = np.argmin(distances[i, :])
    distance = distances[i, j]
    if distance < Th_L2:
        p1 = np.array(sift_kp1[i].pt)
        p2 = np.array(sift_kp2[j].pt)
        p2[0] = p2[0] + image1.shape[1]
        cv2.line(img=image_dual,
                 pt1 = p1.astype(np.int32),
                 pt2 = p2.astype(np.int32),
                 color=[0, 255, 0],
                 thickness=1,
                 lineType=cv2.LINE_AA)
cv2.imshow("dual", image_dual)


cv2.waitKey(0)

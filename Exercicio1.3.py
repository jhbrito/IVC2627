import cv2
import os
from PIL import Image
import numpy as np
import skimage
import matplotlib.pyplot as plt

folder = "Files"
file = "Sharbat_Gula.jpg"

file_path = os.path.join(folder, file)

image_opencv = cv2.imread(file_path)  # load BGR

image_pil = Image.open(file_path)  # load RGB
image_pil_np = np.array(image_pil)

image_skimage = skimage.io.imread(file_path)  # load RGB

cv2.imshow("OpenCV", image_opencv)
cv2.imshow("Pillow", image_pil_np)
cv2.imshow("Skimage", image_skimage)

cv2.imshow("Pillow BGR", cv2.cvtColor(image_pil_np, cv2.COLOR_RGB2BGR))
cv2.imshow("Skimage BGR", cv2.cvtColor(image_skimage, cv2.COLOR_RGB2BGR))

plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(image_opencv)
plt.axis('off')
plt.title("OpenCV (BGR)")

plt.subplot(2, 3, 2)
plt.imshow(image_pil_np)
plt.axis('off')
plt.title("Pillow (RGB)")

plt.subplot(2, 3, 3)
plt.imshow(image_skimage)
plt.axis('off')
plt.title("Skimage (RGB)")

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(image_opencv, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("OpenCV (RGB)")

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(image_pil_np, cv2.COLOR_RGB2BGR))
plt.axis('off')
plt.title("Pillow (BGR)")

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(image_skimage, cv2.COLOR_RGB2BGR))
plt.axis('off')
plt.title("Skimage (BGR)")

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()


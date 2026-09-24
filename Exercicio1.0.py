import time
import numpy as np
import cv2

h = 600
w = 800

imagem = np.zeros((h, w), dtype=np.uint8)
imagem[:, :] = 127
# imagem = imagem + 127

# imagem = np.ones((576, 704), dtype=np.uint8)
# imagem = imagem * 127

imagem[0:10, :] = 0
imagem[:, 0:10] = 0
imagem[:, imagem.shape[1]-10:] = 0
imagem[imagem.shape[0]-10:, :] = 0

cv2.imshow("imagem", imagem)

imagem2 = np.zeros((h, w, 3), dtype=np.uint8)
# imagem2[0:h/2, 0:w/2] = [0, 0, 255]
start = time.time()

imagem2[0:int(h/2), 0:int(w/2), 2] = 255
imagem2[0:int(h/2), int(w/2):, 1] = 255
imagem2[int(h/2):, 0:int(w/2), 0] = 255

imagem2[int(h/2):, int(w/2):, 1] = 255
imagem2[int(h/2):, int(w/2):, 2] = 255
end = time.time()
print("Elapsed time Numpy = %s" % (end - start))

cv2.imshow("imagem2", imagem2)

imagem3 = np.zeros((h, w, 3), dtype=np.uint8)
start = time.time()
for y in range(int(h/2)):
    for x in range(int(w/2)):
        imagem3[y, x, 2] = 255

for y in range(int(h/2)):
    for x in range(int(w/2), w):
        imagem3[y, x, 1] = 255

for y in range(int(h/2), h):
    for x in range(int(w/2)):
        imagem3[y, x, 0] = 255

for y in range(int(h/2), h):
    for x in range(int(w / 2), w):
        imagem3[y, x, 1] = 255
        imagem3[y, x, 2] = 255
end = time.time()
print("Elapsed time For = %s" % (end - start))

cv2.imshow("imagem3", imagem3)

imagem4 = imagem2
imagem4 = imagem2 / 255.0
cv2.imshow("imagem4", imagem4)

imagem5 = np.ones((h, w, 3), dtype=np.uint8)
imagem5 = imagem5 * 1.0

cv2.imshow("imagem5", imagem5)

cv2.waitKey(0)


cv2.destroyAllWindows()

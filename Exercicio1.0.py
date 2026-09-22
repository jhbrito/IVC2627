from email.mime import image

import numpy as np
import cv2

h = 576
w = 704

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
imagem2[0:int(h/2), 0:int(w/2), 2] = 255
imagem2[0:int(h/2), int(w/2):, 1] = 255
imagem2[int(h/2):, 0:int(w/2), 0] = 255

imagem2[int(h/2):, int(w/2):, 1] = 255
imagem2[int(h/2):, int(w/2):, 2] = 255

cv2.imshow("imagem2", imagem2)

cv2.waitKey(0)
cv2.destroyAllWindows()

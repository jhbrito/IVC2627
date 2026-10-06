import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.ndimage import histogram

folder = "Files"
files = ["baboon.png", "cao.jpg", "lena.png", "Sharbat_Gula.jpg"]

def ivc_gray_pdf(image_gray):
    # histogram = cv2.calcHist(image_gray,
    #                          [0],
    #                          None,
    #                          [256],
    #                          [0, 256])
    # pdf = histogram / (image_gray.shape[0] * image_gray.shape[1])

    histogram = np.zeros((256,), dtype=np.uint32)
    # for i in range(256):
    #     for y in range(image_gray.shape[0]):
    #         for x in range(image_gray.shape[1]):
    #             if image_gray[y, x] == i:
    #                 histogram[i] = histogram[i] + 1
    for i in range(256):
        histogram[i] = np.sum(image_gray == i)

    pdf = histogram / (image_gray.shape[0] * image_gray.shape[1])
    return pdf


def ivc_pdf_2_cdf(pdf):
    cdf = np.zeros(pdf.shape, dtype=pdf.dtype)
    cdf[0] = pdf[0]
    for i in range(1, pdf.shape[0]):
        cdf[i] = cdf[i - 1] + pdf[i]
    return cdf

def ivc_equalize_image(image_gray, cdf):
    cdfmin = cdf[0]
    g = np.zeros(image_gray.shape, dtype=image_gray.dtype)
    for y in range(image_gray.shape[0]):
        for x in range(image_gray.shape[1]):
            g[y, x] = 255 * ((cdf[image_gray[y, x]] - cdfmin) / (1 - cdfmin))
    return g


# iterate files in the list
for i, file in enumerate(files):
    image = cv2.imread(os.path.join(folder, file))
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # calcular pdf
    # pdf = ivc_gray_pdf(image_gray)
    V = image_hsv[:, :, 2]
    pdf = ivc_gray_pdf(V)
    # calcular cdf
    cdf = ivc_pdf_2_cdf(pdf)

    # gerar imagem equalizada
    g = ivc_equalize_image(V, cdf)
    image_HSV_equalized = image_hsv.copy()
    image_HSV_equalized[:,:,2] = g

    # calcular pdf equalizada
    pdf_equalized = ivc_gray_pdf(g)
    # calcular cdf equalizada
    cdf_equalized = ivc_pdf_2_cdf(pdf_equalized)

    plt.subplot(len(files), 7, i*7+1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(file)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(len(files), 7, i * 7 + 2)
    plt.imshow(cv2.cvtColor(V, cv2.COLOR_GRAY2RGB))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(len(files), 7, i * 7 + 3)
    plt.bar(range(256), pdf)
    plt.title("pdf")

    plt.subplot(len(files), 7, i * 7 + 4)
    plt.plot(range(256), cdf)
    plt.title("cdf")

    plt.subplot(len(files), 7, i * 7 + 5)
    plt.imshow(cv2.cvtColor(image_HSV_equalized, cv2.COLOR_HSV2RGB))
    plt.xticks([])
    plt.yticks([])

    plt.subplot(len(files), 7, i * 7 + 6)
    plt.bar(range(256), pdf_equalized)
    plt.title("pdf equalized")

    plt.subplot(len(files), 7, i * 7 + 7)
    plt.plot(range(256), cdf_equalized)
    plt.title("cdf equalized")

plt.show()




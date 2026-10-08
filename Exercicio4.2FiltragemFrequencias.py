import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from networkx.drawing import layout

folder = "Files"
files = ["riscas horizontais", "riscas verticais", "quadrado",
         "baboon.png", "cao.jpg", "lena.png", "Sharbat_Gula.jpg",
         "moedas.jpg"]

for i, file in enumerate(files):
    if file == "riscas horizontais":
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        image[0:32, :, :] = 255
        image[64:96, :, :] = 255
        image[128:160, :, :] = 255
        image[192:224, :, :] = 255
    elif file == "riscas verticais":
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        image[:, 0:32, :] = 255
        image[:, 64:96, :] = 255
        image[:, 128:160, :] = 255
        image[:, 192:224, :] = 255
    elif file == "quadrado":
        image = np.zeros((256, 256, 3), dtype=np.uint8)
        image[64:128, 64:128, :] = 255
    else:
        image = cv2.imread(os.path.join(folder, file))

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = (image_gray / 255.0).astype(np.float32)

    image_fft = np.fft.fft2(image_gray)

    image_fft_show = np.abs(image_fft)
    image_fft_show = image_fft_show / np.mean(image_fft_show)

    image_fft_shift = np.fft.fftshift(image_fft)

    image_fft_shift_show = np.abs(image_fft_shift)
    image_fft_shift_show = image_fft_shift_show / np.mean(image_fft_shift_show)

    filter_low_pass = np.zeros(image_fft_shift.shape, dtype=np.float32)
    centre_x = filter_low_pass.shape[1] / 2
    centre_y = filter_low_pass.shape[0] / 2
    radius = np.min(filter_low_pass.shape)/8

    for y in range(filter_low_pass.shape[0]):
        for x in range(filter_low_pass.shape[1]):
            d = np.sqrt((x - centre_x)**2 + (y - centre_y)**2)
            if d < radius:
                filter_low_pass[y, x] = 1

    image_fft_shift_filtered = image_fft_shift * filter_low_pass

    image_fft_shift_filtered_show = np.abs(image_fft_shift_filtered)
    image_fft_shift_filtered_show = image_fft_shift_filtered_show / np.mean(image_fft_shift_filtered_show)

    image_fft_shift_filtered_unshift = np.fft.ifftshift(image_fft_shift_filtered)

    image_fft_shift_filtered_unshift_show = np.abs(image_fft_shift_filtered_unshift)
    image_fft_shift_filtered_unshift_show = image_fft_shift_filtered_unshift_show / np.mean(image_fft_shift_filtered_unshift_show)

    image_fft_shift_filtered_unshift_ifft = np.fft.ifft2(image_fft_shift_filtered_unshift)
    image_fft_shift_filtered_unshift_ifft = np.abs(image_fft_shift_filtered_unshift_ifft)

    plt.subplots(dpi=300, layout="constrained")
    plt.axis('off')

    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')

    plt.subplot(3, 3, 4)
    plt.imshow(cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB))
    plt.title("Grayscale")
    plt.axis('off')

    plt.subplot(3, 3, 5)
    plt.imshow(cv2.cvtColor(image_fft_show, cv2.COLOR_GRAY2RGB))
    plt.title("FFT")
    plt.axis('off')

    plt.subplot(3, 3, 6)
    plt.imshow(cv2.cvtColor(image_fft_shift_show, cv2.COLOR_GRAY2RGB))
    plt.title("FFT Shift")
    plt.axis('off')

    plt.subplot(3, 3, 3)
    plt.imshow(cv2.cvtColor(filter_low_pass, cv2.COLOR_GRAY2RGB))
    plt.title("Filter Low Pass")
    plt.axis('off')

    plt.subplot(3, 3, 9)
    plt.imshow(cv2.cvtColor(image_fft_shift_filtered_show, cv2.COLOR_GRAY2RGB))
    plt.title("FFT Shift Filtered")
    plt.axis('off')

    plt.subplot(3, 3, 8)
    plt.imshow(cv2.cvtColor(image_fft_shift_filtered_unshift_show, cv2.COLOR_GRAY2RGB))
    plt.title("FFT Shift Filtered Unshift")
    plt.axis('off')

    plt.subplot(3, 3, 7)
    plt.imshow(cv2.cvtColor(image_fft_shift_filtered_unshift_ifft, cv2.COLOR_GRAY2RGB))
    plt.title("Filtered image")
    plt.axis('off')

    plt.show()


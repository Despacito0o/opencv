import cv2
import numpy as np
import matplotlib.pyplot as plt
def create_filters(rows, cols, D0=60):
    ilpf = np.zeros((rows, cols))
    cv2.circle(ilpf, (cols // 2, rows // 2), D0, 1, -1)
    ihpf = 1 - ilpf
    return ilpf, ihpf
image_path = r'C:\Users\Administrator\Desktop\1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("无法读取图像，请检查图像路径。")
else:
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    rows, cols = image.shape
    ilpf, ihpf = create_filters(rows, cols)
    fshift_ilpf = fshift * ilpf
    fshift_ihpf = fshift * ihpf
    f_ilpf = np.fft.ifftshift(fshift_ilpf)
    img_back_ilpf = np.fft.ifft2(f_ilpf)
    img_back_ilpf = np.abs(img_back_ilpf)
    f_ihpf = np.fft.ifftshift(fshift_ihpf)
    img_back_ihpf = np.fft.ifft2(f_ihpf)
    img_back_ihpf = np.abs(img_back_ihpf)
    plt.subplot(131), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img_back_ilpf, cmap='gray')
    plt.title('Ideal Low Pass Filtered'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(img_back_ihpf, cmap='gray')
    plt.title('Ideal High Pass Filtered'), plt.xticks([]), plt.yticks([])
    plt.show()
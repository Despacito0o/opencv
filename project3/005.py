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
    gamma_l = 0.5  # 低频增益(控制光照)
    gamma_h = 1.5  # 高频增益(控制细节)
    D0 = 30  # 截止频率
    c = 1  # 斜率控制
    img_float = np.float32(image) + 1e-6
    log_img = np.log(img_float)
    dft = np.fft.fft2(log_img)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img_float.shape
    u, v = np.mgrid[-rows // 2:rows // 2, -cols // 2:cols // 2]
    D = np.sqrt(u ** 2 + v ** 2)
    H = (gamma_h - gamma_l) * (1 - np.exp(-c * (D ** 2) / (D0 ** 2))) + gamma_l
    filtered = dft_shift * H
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.exp(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 显示原始图像、理想低通滤波图像、理想高通滤波图像和同态滤波图像
    plt.subplot(221), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(img_back_ilpf, cmap='gray')
    plt.title('Ideal Low Pass Filtered'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(img_back_ihpf, cmap='gray')
    plt.title('Ideal High Pass Filtered'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(img_back, cmap='gray')
    plt.title('Homomorphic Filtered'), plt.xticks([]), plt.yticks([])
    plt.show()
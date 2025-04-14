import cv2
import numpy as np
# 图片路径
image_path = r'C:\Users\Administrator\Desktop\1.jpg'
# 读取图片
image = cv2.imread(image_path)
# 检查图像是否成功读取
if image is None:
    print("无法读取图像，请检查图像路径。")
else:
    # 使用3x3和5x5的均值滤波器模糊图像
    blurred_3x3_mean = cv2.blur(image, (3, 3))
    blurred_5x5_mean = cv2.blur(image, (5, 5))
    # 使用核大小为3和5的中值滤波器模糊图像
    blurred_3_median = cv2.medianBlur(image, 3)
    blurred_5_median = cv2.medianBlur(image, 5)
    # 转换为灰度图像，因为Sobel算子通常在单通道图像上操作
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 使用Sobel算子检测图像边缘
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobelx ** 2 + sobely ** 2)
    edges = np.uint8(edges / edges.max() * 255)
    # 显示原始图像、模糊后的图像和边缘检测后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('3x3 Mean Blurred Image', blurred_3x3_mean)
    cv2.imshow('5x5 Mean Blurred Image', blurred_5x5_mean)
    cv2.imshow('3 Median Blurred Image', blurred_3_median)
    cv2.imshow('5 Median Blurred Image', blurred_5_median)
    cv2.imshow('Sobel Edge Detection', edges)
    # 等待按键退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()
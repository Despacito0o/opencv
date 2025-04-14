import cv2
import numpy as np
# 读取图像
image_path = r'C:\Users\Administrator\Desktop\1.jpg'
image = cv2.imread(image_path)
# 检查图像是否成功读取
if image is None:
    print("无法读取图像，请检查图像路径是否正确。")
else:
    # 进行水平镜像
    mirrored_image = cv2.flip(image, 1)
    # 将镜像后的图像缩小为原来的一半大小，使用双线性插值
    height, width = mirrored_image.shape[:2]
    shrunk_image = cv2.resize(mirrored_image, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
    # 逆时针旋转45度，使用双三次插值
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 45, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_CUBIC)
    # 对旋转后的图像进行转置操作
    transposed_image = cv2.transpose(rotated_image)
    # 显示原始图像、处理后的图像、旋转后的图像和转置后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Processed Image', shrunk_image)
    cv2.imshow('Rotated Image', rotated_image)
    cv2.imshow('Transposed Image', transposed_image)
    # 等待按键关闭窗口
    cv2.waitKey(0)
cv2.destroyAllWindows()
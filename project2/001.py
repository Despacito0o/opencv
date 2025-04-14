import cv2
import numpy as np
# 读取图像
image_path = r'C:\Users\Administrator\Desktop\1.jpg'
image = cv2.imread(image_path)
# 检查图像是否成功读取
if image is None:
    print("无法读取图像，请检查图像路径是否正确。")
else:
    # 定义平移矩阵
    translation_matrix = np.float32([[1, 0, 50], [0, 1, 30]])
    # 获取图像的高度和宽度
    height, width = image.shape[:2]
    # 执行平移操作，使用最近邻插值法
    translated_image = cv2.warpAffine(image, translation_matrix, (width, height), flags=cv2.INTER_NEAREST)
    # 显示原始图像和平移后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Translated Image', translated_image)
    # 等待按键关闭窗口
    cv2.waitKey(0)
cv2.destroyAllWindows()
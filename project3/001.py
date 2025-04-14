import cv2
import numpy as np
# 读取图像
image_path = r'C:\Users\Administrator\Desktop\1.jpg'
image = cv2.imread(image_path)
# 检查图像是否成功读取
if image is None:
    print("无法读取图像，请检查图像路径。")
else:
    # 线性变换增强亮度
    enhanced_image = np.clip(image * 1.5, 0, 255).astype(np.uint8)
    # 显示原始图像和增强后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Enhanced Image', enhanced_image)
    # 等待按键退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()
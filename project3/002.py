import cv2
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
    # 显示原始图像和模糊后的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('3x3 Mean Blurred Image', blurred_3x3_mean)
    cv2.imshow('5x5 Mean Blurred Image', blurred_5x5_mean)
    cv2.imshow('3 Median Blurred Image', blurred_3_median)
    cv2.imshow('5 Median Blurred Image', blurred_5_median)
    # 等待按键退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()
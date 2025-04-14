import cv2

# 读取图像
img = cv2.imread(r'C:\Users\Administrator\Desktop\1.jpg')
if img is None:
    print("错误：无法读取图像，请检查路径")
    exit()
# 实验（3）：缩放并裁剪中心区域
height, width, channels = img.shape
# 缩放至原尺寸的一半
scaled_img = cv2.resize(img, (width // 2, height // 2))
h_scaled, w_scaled = scaled_img.shape[:2]
# 计算中心区域坐标
y_start = (h_scaled - 100) // 2
y_end = y_start + 100
x_start = (w_scaled - 100) // 2
x_end = x_start + 100
# 确保裁剪区域有效
if y_start >= 0 and x_start >= 0 and y_end <= h_scaled and x_end <= w_scaled:
    crop_img = scaled_img[y_start:y_end, x_start:x_end]
    cv2.imshow('Scaled and Cropped', crop_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("警告：缩放后图像尺寸过小，无法裁剪100x100区域")

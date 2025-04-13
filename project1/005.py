import cv2

# 读取图像
img = cv2.imread(r'C:\Users\Administrator\Desktop\1.jpg')
if img is None:
    print("错误：无法读取图像，请检查路径")
    exit()

# 实验（5）：图像翻转与旋转
# 水平翻转
flip_h = cv2.flip(img, 1)
# 垂直翻转
flip_v = cv2.flip(img, 0)
# 顺时针旋转90度
rot_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

cv2.imshow('Horizontal Flip', flip_h)
cv2.imshow('Vertical Flip', flip_v)
cv2.imshow('Rotated 90°', rot_90)
cv2.waitKey(0)
cv2.destroyAllWindows()
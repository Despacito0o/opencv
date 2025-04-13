import cv2

# 读取图像
img = cv2.imread(r'C:\Users\Administrator\Desktop\1.jpg')
if img is None:
    print("错误：无法读取图像，请检查路径")
    exit()

# 实验（4）：转换为HSV并分离显示通道
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_img)
cv2.imshow('H Channel', h)
cv2.imshow('S Channel', s)
cv2.imshow('V Channel', v)
cv2.waitKey(0)
cv2.destroyAllWindows()

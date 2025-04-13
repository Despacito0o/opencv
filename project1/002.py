import cv2

# 读取图像
img = cv2.imread(r'C:\Users\Administrator\Desktop\1.jpg')
if img is None:
    print("错误：无法读取图像，请检查路径")
    exit()
# 实验（2）：转换为灰度图并保存
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_image.jpg', gray_img)
cv2.imwrite('gray_image.png', gray_img)
cv2.imwrite('gray_image.bmp', gray_img)
cv2.imshow('gray_image.jpg',gray_img)
cv2.imshow('gray_image.png',gray_img)
cv2.imshow('gray_image.bmp',gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

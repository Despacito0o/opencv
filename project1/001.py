import cv2

# 读取图像
img = cv2.imread(r'C:\Users\Administrator\Desktop\1.jpg')
if img is None:
    print("错误：无法读取图像，请检查路径")
    exit()

# 实验（1）：输出图像基本信息
print("图像类型:", type(img))
height, width, channels = img.shape
print("宽度:", width)
print("高度:", height)
print("通道数:", channels)
print("数据类型:", img.dtype)
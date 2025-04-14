# OpenCV图像增强实战教程：从理论到代码实现 🔥🚀

> 📚 想要掌握图像增强的核心技术？本文手把手教你使用OpenCV实现多种图像增强技术，从基础的线性变换到高级的频域滤波，全方位提升你的图像处理能力！适合初学者和进阶开发者！

## 目录
- [1. 线性变换：调整图像亮度](#1-线性变换调整图像亮度)
- [2. 空间域滤波：均值滤波与中值滤波](#2-空间域滤波均值滤波与中值滤波)
- [3. 边缘检测：Sobel算子实现](#3-边缘检测sobel算子实现)
- [4. 频域滤波：理想低通与高通滤波器](#4-频域滤波理想低通与高通滤波器)
- [5. 高级应用：同态滤波处理光照不均](#5-高级应用同态滤波处理光照不均)
- [6. 直方图均衡化：提升图像对比度](#6-直方图均衡化提升图像对比度)
- [7. 图像锐化：拉普拉斯算子应用](#7-图像锐化拉普拉斯算子应用)
- [8. 最佳实践与常见问题](#8-最佳实践与常见问题)

> 🌟 本教程所有代码已开源至GitHub: [https://github.com/Despacito0o/opencv](https://github.com/Despacito0o/opencv)，欢迎Star和Fork！

## 1. 线性变换：调整图像亮度

线性变换是最基础的图像增强方法，通过简单的乘法运算就能调整图像的亮度。这种技术在图像预处理中非常常用，可以快速调整图像整体的明暗程度。

### 核心原理
图像亮度调整的公式：`g(x,y) = α·f(x,y)`，其中α为增益系数。

- 当α > 1时，图像整体变亮
- 当α < 1时，图像整体变暗
- 当α = 1时，图像保持不变

除了乘法操作，还可以使用加法进行亮度调整：`g(x,y) = f(x,y) + β`，其中β为亮度偏移量。

### 代码实现

```python
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
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/11f7a1bbdb2c4b4facd16005337d2850.png)


> 💡 **技巧**：使用`np.clip()`函数防止像素值超出[0,255]范围，确保增强后的图像不会溢出或失真。当乘以较大系数时尤其重要！

### 扩展：对比度调整

除了简单的亮度调整，我们还可以同时调整亮度和对比度：

```python
def adjust_brightness_contrast(image, alpha=1.0, beta=0):
    """
    调整图像的亮度和对比度
    alpha: 对比度调整因子 (>1增加对比度，<1降低对比度)
    beta: 亮度调整因子 (>0增加亮度，<0降低亮度)
    """
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

# 使用示例
brightened = adjust_brightness_contrast(image, alpha=1.0, beta=50)  # 只增加亮度
contrasted = adjust_brightness_contrast(image, alpha=1.5, beta=0)   # 只增加对比度
both_adjusted = adjust_brightness_contrast(image, alpha=1.3, beta=30)  # 同时调整
```

## 2. 空间域滤波：均值滤波与中值滤波

空间域滤波是通过像素邻域操作处理图像的方法，常用于去噪和平滑。空间域滤波使用卷积核在整个图像上进行操作，不同的卷积核可以实现不同的效果。

### 均值滤波 vs 中值滤波
- **均值滤波**：取邻域像素的平均值，对高斯噪声效果较好，但会模糊边缘
- **中值滤波**：取邻域像素的中值，对椒盐噪声效果显著，能更好地保留边缘

### 滤波核大小选择指南
- 3×3：轻微平滑，保留大部分细节
- 5×5：中等平滑，去除更多噪声但会丢失一些细节
- 7×7及以上：强烈平滑，适合噪声严重的情况

### 代码实现

```python
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
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1409b57f1e4b40a394f4aa465a543bed.png)


> 🔍 **观察**：滤波核尺寸越大，平滑效果越明显，但边缘保持能力越差。实际应用中需要权衡平滑度和细节保留。

### 扩展：高斯滤波

均值滤波给予窗口内每个像素相同的权重，而高斯滤波则给予中心像素更高的权重，边缘像素权重逐渐减小，这样可以获得更自然的平滑效果：

```python
# 高斯滤波示例
blurred_gaussian_3 = cv2.GaussianBlur(image, (3, 3), 0)
blurred_gaussian_5 = cv2.GaussianBlur(image, (5, 5), 0)

# 参数说明：
# - 第一个参数：输入图像
# - 第二个参数：高斯核大小，必须是奇数
# - 第三个参数：x方向的标准差，0表示根据核大小自动计算
```

## 3. 边缘检测：Sobel算子实现

边缘检测是提取图像中物体轮廓的重要技术，能够帮助我们识别图像中的物体边界和形状特征。Sobel算子是一种常用的一阶微分算子，可以有效地检测图像中的边缘。

### Sobel算子原理
Sobel算子计算图像在水平方向(x)和垂直方向(y)的梯度，通过梯度的平方和的平方根得到边缘强度。

水平方向Sobel算子：
```
-1  0  1
-2  0  2
-1  0  1
```

垂直方向Sobel算子：
```
-1 -2 -1
 0  0  0
 1  2  1
```

### 代码实现

```python
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
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fef61b716f4f4dba9feadb39bdd402bb.png)

> ⚠️ **注意**：边缘检测前先转换为灰度图像，可以减少计算量并避免颜色通道间的干扰。图像噪声会导致边缘检测结果不理想，建议先进行适当的滤波处理。

### 扩展：Canny边缘检测

Canny边缘检测是更加高级的边缘检测算法，包含多个步骤：高斯滤波、梯度计算、非极大值抑制和双阈值检测。相比Sobel，它能提供更加清晰和完整的边缘：

```python
# Canny边缘检测示例
edges_canny = cv2.Canny(gray, 100, 200)  # 参数为低阈值和高阈值

# 参数调优技巧:
# - 低阈值:高阈值比例通常为1:2或1:3
# - 较低的阈值会检测出更多边缘，但可能包含噪声
# - 较高的阈值会减少检测的边缘，但边缘会更加可靠
```

## 4. 频域滤波：理想低通与高通滤波器

频域滤波是通过傅里叶变换将图像从空间域转换到频域进行处理的方法。在频域中，不同频率成分代表图像中不同的信息：低频成分代表图像的整体结构和轮廓，高频成分代表图像的细节和边缘。

### 低通滤波 vs 高通滤波
- **低通滤波**：保留低频信息（图像的轮廓和大的结构），去除高频部分（细节和噪声），结果是平滑的图像
- **高通滤波**：保留高频信息（图像的边缘和细节），去除低频部分（平滑区域），结果是突出边缘的图像

### 频域滤波原理
1. 使用傅里叶变换将图像从空间域转换到频域
2. 在频域应用滤波器
3. 使用逆傅里叶变换将处理后的图像变换回空间域

### 代码实现

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
def create_filters(rows, cols, D0=60):
    ilpf = np.zeros((rows, cols))
    cv2.circle(ilpf, (cols // 2, rows // 2), D0, 1, -1)
    ihpf = 1 - ilpf
    return ilpf, ihpf
image_path = r'C:\Users\Administrator\Desktop\1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("无法读取图像，请检查图像路径。")
else:
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    rows, cols = image.shape
    ilpf, ihpf = create_filters(rows, cols)
    fshift_ilpf = fshift * ilpf
    fshift_ihpf = fshift * ihpf
    f_ilpf = np.fft.ifftshift(fshift_ilpf)
    img_back_ilpf = np.fft.ifft2(f_ilpf)
    img_back_ilpf = np.abs(img_back_ilpf)
    f_ihpf = np.fft.ifftshift(fshift_ihpf)
    img_back_ihpf = np.fft.ifft2(f_ihpf)
    img_back_ihpf = np.abs(img_back_ihpf)
    plt.subplot(131), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132), plt.imshow(img_back_ilpf, cmap='gray')
    plt.title('Ideal Low Pass Filtered'), plt.xticks([]), plt.yticks([])
    plt.subplot(133), plt.imshow(img_back_ihpf, cmap='gray')
    plt.title('Ideal High Pass Filtered'), plt.xticks([]), plt.yticks([])
    plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4da2b41411264a00b7dbd8857a9c5c62.png)


> 🚀 **进阶技巧**：理想低通/高通滤波器在频域边界会产生明显的振铃效应（边缘处出现波纹状伪影），实际应用中可以考虑使用高斯或巴特沃斯滤波器来获得更自然的效果。

### 扩展：可视化傅里叶频谱

频谱可视化能帮助我们理解图像的频域特性：

```python
# 显示频谱
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

# 频谱解读:
# - 中心点代表零频率(DC)分量
# - 亮度越高表示该频率分量越强
# - 图像的主要结构通常集中在中心区域(低频)
# - 噪声和细节通常分布在远离中心的区域(高频)
```

## 5. 高级应用：同态滤波处理光照不均

同态滤波是一种非线性滤波技术，可以同时增强图像细节和平衡光照不均。这种技术在医学图像处理、天文图像增强和不均匀光照条件下的图像处理中特别有用。

### 同态滤波原理
同态滤波基于照明-反射模型，将图像分解为照明成分（光照，低频）和反射成分（物体表面特性，高频）：
1. 通过对数变换将乘法关系转换为加法关系
2. 对低频照明部分减弱，对高频反射部分增强
3. 通过指数变换恢复原始域

### 代码实现

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
def create_filters(rows, cols, D0=60):
    ilpf = np.zeros((rows, cols))
    cv2.circle(ilpf, (cols // 2, rows // 2), D0, 1, -1)
    ihpf = 1 - ilpf
    return ilpf, ihpf
image_path = r'C:\Users\Administrator\Desktop\1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    print("无法读取图像，请检查图像路径。")
else:
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    rows, cols = image.shape
    ilpf, ihpf = create_filters(rows, cols)
    fshift_ilpf = fshift * ilpf
    fshift_ihpf = fshift * ihpf
    f_ilpf = np.fft.ifftshift(fshift_ilpf)
    img_back_ilpf = np.fft.ifft2(f_ilpf)
    img_back_ilpf = np.abs(img_back_ilpf)
    f_ihpf = np.fft.ifftshift(fshift_ihpf)
    img_back_ihpf = np.fft.ifft2(f_ihpf)
    img_back_ihpf = np.abs(img_back_ihpf)
    gamma_l = 0.5  # 低频增益(控制光照)
    gamma_h = 1.5  # 高频增益(控制细节)
    D0 = 30  # 截止频率
    c = 1  # 斜率控制
    img_float = np.float32(image) + 1e-6
    log_img = np.log(img_float)
    dft = np.fft.fft2(log_img)
    dft_shift = np.fft.fftshift(dft)
    rows, cols = img_float.shape
    u, v = np.mgrid[-rows // 2:rows // 2, -cols // 2:cols // 2]
    D = np.sqrt(u ** 2 + v ** 2)
    H = (gamma_h - gamma_l) * (1 - np.exp(-c * (D ** 2) / (D0 ** 2))) + gamma_l
    filtered = dft_shift * H
    f_ishift = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.exp(img_back)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 显示原始图像、理想低通滤波图像、理想高通滤波图像和同态滤波图像
    plt.subplot(221), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(img_back_ilpf, cmap='gray')
    plt.title('Ideal Low Pass Filtered'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(img_back_ihpf, cmap='gray')
    plt.title('Ideal High Pass Filtered'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(img_back, cmap='gray')
    plt.title('Homomorphic Filtered'), plt.xticks([]), plt.yticks([])
    plt.show()
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/be0b989001ac4fce953e862b1c37b665.png)

> 🔧 **参数调优**：
> - `gamma_l`：控制低频增益，值小于1可以压制光照变化，改善光照不均
> - `gamma_h`：控制高频增益，值大于1可以增强图像细节和边缘
> - `D0`：截止频率，控制低频和高频的分界点
> - `c`：滤波函数的陡峭度，值越大变化越剧烈

### 同态滤波的实际应用场景
- 医学图像增强：提高X光片、CT扫描和MRI图像的可读性
- 夜间或阴影下拍摄的照片修复
- 卫星和航空图像处理：平衡不同区域的光照差异

## 6. 直方图均衡化：提升图像对比度

直方图均衡化是一种简单但非常有效的图像增强技术，通过重新分配图像的灰度值分布，使图像具有更均匀的灰度分布，从而提高图像的对比度。这对于对比度较低、过亮或过暗的图像尤其有效。

### 直方图均衡化原理
1. 计算图像的灰度直方图
2. 计算累积分布函数(CDF)
3. 将CDF值归一化到[0,255]范围，作为灰度映射函数
4. 根据映射函数对原图像进行变换

### 代码实现

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image_path = r'C:\Users\Administrator\Desktop\1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("无法读取图像，请检查图像路径。")
else:
    # 应用直方图均衡化
    equalized = cv2.equalizeHist(image)
    
    # 创建并绘制直方图
    hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])
    
    # 显示结果
    plt.figure(figsize=(12, 8))
    
    plt.subplot(221), plt.imshow(image, cmap='gray')
    plt.title('原始图像'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(222), plt.imshow(equalized, cmap='gray')
    plt.title('均衡化后图像'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(223), plt.plot(hist_original)
    plt.title('原始直方图'), plt.xlim([0, 256])
    
    plt.subplot(224), plt.plot(hist_equalized)
    plt.title('均衡化后直方图'), plt.xlim([0, 256])
    
    plt.tight_layout()
    plt.show()
```

### 自适应直方图均衡化 (CLAHE)
标准直方图均衡化应用于整个图像，可能会导致某些区域的对比度过度增强。CLAHE方法通过在图像的小区域上应用直方图均衡化，然后使用双线性插值将相邻区域平滑地融合，可以获得更自然的增强效果：

```python
# 应用CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_image = clahe.apply(image)

# 参数说明:
# - clipLimit: 对比度限制阈值，防止噪声放大
# - tileGridSize: 分块大小，较小的块会增强局部细节
```

## 7. 图像锐化：拉普拉斯算子应用

图像锐化是增强图像中边缘和细节的过程，能够使模糊的图像变得更加清晰。拉普拉斯算子是一种常用的图像锐化工具，它通过检测图像中的二阶导数来识别亮度快速变化的区域。

### 拉普拉斯算子原理
拉普拉斯算子是一种二阶微分算子，常用的3×3拉普拉斯卷积核有：

```
 0  1  0
 1 -4  1
 0  1  0
```

或者：

```
 1  1  1
 1 -8  1
 1  1  1
```

### 图像锐化的实现

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image_path = r'C:\Users\Administrator\Desktop\1.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("无法读取图像，请检查图像路径。")
else:
    # 使用拉普拉斯算子
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    
    # 转换为可显示的格式
    laplacian = np.uint8(np.absolute(laplacian))
    
    # 锐化图像 = 原图 + 拉普拉斯变换结果
    sharpened = cv2.addWeighted(image, 1.5, laplacian, -0.5, 0)
    
    # 显示结果
    plt.figure(figsize=(12, 4))
    
    plt.subplot(131), plt.imshow(image, cmap='gray')
    plt.title('原始图像'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(132), plt.imshow(laplacian, cmap='gray')
    plt.title('拉普拉斯变换'), plt.xticks([]), plt.yticks([])
    
    plt.subplot(133), plt.imshow(sharpened, cmap='gray')
    plt.title('锐化后的图像'), plt.xticks([]), plt.yticks([])
    
    plt.tight_layout()
    plt.show()
```

### USM锐化(Unsharp Masking)
USM是一种更高级的锐化技术，可以提供更自然的锐化效果：

```python
# USM锐化
blurred = cv2.GaussianBlur(image, (5, 5), 0)
usm_sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

# 原理：
# 1. 对原图进行高斯模糊
# 2. 从原图中减去模糊图像，得到边缘信息
# 3. 将边缘信息加回原图，增强边缘
```

## 8. 最佳实践与常见问题

### 选择合适的图像增强方法

| 图像问题 | 推荐方法 |
|---------|--------|
| 亮度过低/过高 | 线性变换、直方图均衡化 |
| 对比度不足 | 直方图均衡化、伽马变换 |
| 噪声干扰 | 均值滤波(高斯噪声)、中值滤波(椒盐噪声) |
| 图像模糊 | 锐化滤波、USM锐化 |
| 光照不均 | 同态滤波 |
| 需要检测边缘 | Sobel算子、Canny边缘检测 |

### 图像增强的常见陷阱
1. **过度增强**：参数设置过高可能导致伪影和噪声放大
2. **细节丢失**：过度平滑会导致重要细节丢失
3. **颜色失真**：在彩色图像上直接应用某些增强方法可能导致颜色失真

### 性能优化技巧
1. **图像尺寸**：处理前先调整图像尺寸，可以显著提高处理速度
2. **并行处理**：使用NumPy的矢量化操作代替循环
3. **GPU加速**：对于大规模图像处理，考虑使用OpenCV的CUDA模块

```python
# 使用OpenCV的CUDA模块示例(需要CUDA支持)
import cv2

# 检查CUDA是否可用
print("CUDA可用：", cv2.cuda.getCudaEnabledDeviceCount() > 0)

# 使用GPU加速的高斯模糊
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    gpu_image = cv2.cuda_GpuMat()
    gpu_image.upload(image)
    gpu_result = cv2.cuda.createGaussianFilter(
        cv2.CV_8UC1, cv2.CV_8UC1, (5, 5), 0)
    gpu_blurred = gpu_result.apply(gpu_image)
    blurred = gpu_blurred.download()
```

## 更多资源

想要深入学习OpenCV和图像处理？以下资源将助你快速提升：

- 🌟 **完整项目代码**：[GitHub - Despacito0o/opencv](https://github.com/Despacito0o/opencv)
- 📚 **更多图像处理示例**：
  - [基础操作实验](https://github.com/Despacito0o/opencv/tree/main/project1)
  - [几何变换与频域分析](https://github.com/Despacito0o/opencv/tree/main/project2)
  - [图像增强实战](https://github.com/Despacito0o/opencv/tree/main/project3)
- 📝 **环境配置指南**：

```bash
pip install opencv-python==4.11.0
pip install opencv-contrib-python==4.11.0  # 包含扩展模块
pip install matplotlib
pip install numpy
```

### 进阶学习路径
1. **基础**：图像IO、色彩空间转换、基本变换
2. **中级**：滤波器、形态学操作、特征检测
3. **高级**：频域分析、机器学习结合、深度学习结合

> 💼 **实战建议**：将不同的图像增强技术组合使用往往能获得更好的效果。例如，先进行噪声滤波，再进行锐化和对比度增强，最后进行色彩校正。

---

如果这篇教程对你有帮助，请在GitHub上给我的项目点个⭐Star吧！也欢迎通过Issues或PR分享你的想法和改进建议！

[👉 前往GitHub仓库：https://github.com/Despacito0o/opencv](https://github.com/Despacito0o/opencv)
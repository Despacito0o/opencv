# OpenCV图像处理进阶：几何变换与频域分析

## 学习内容
1. 掌握图像几何变换技术，学习图像平移的实现方法及边界处理，理解不同插值方法（最近邻、双线性、双三次）的特点与应用场景。
2. 学习图像缩放的实现过程，掌握图像尺寸变换中的质量控制。
3. 掌握频域变换技术，学习二维离散傅里叶变换的原理与应用。理解离散余弦变换，学习二维离散余弦变换的原理，掌握从变换系数重建图像的方法。

## 实验内容

本项目包含以下实验：
1. **图像平移操作**：实现图像向右平移50像素，向下平移30像素，并使用最近邻插值法处理空白区域。
2. **图像水平镜像与缩放**：对图像进行水平镜像，然后将镜像后的图像缩小为原来的一半大小（使用双线性插值）。
3. **图像旋转与转置**：将图像逆时针旋转45度（使用双三次插值），然后对旋转后的图像进行转置操作。
4. **图像剪切与频域变换**：剪切图像中心区域（大小为原图的1/4），对剪切后的图像进行二维离散傅里叶变换，并显示幅度谱。
5. **二维离散余弦变换与逆变换**：对图像进行二维离散余弦变换，并通过二维离散余弦逆变换复原图像。

## 代码说明

- `001.py` - 图像平移操作示例
- `002.py` - 图像水平镜像与缩放示例
- `003.py` - 图像旋转与转置示例
- `004.py` - 图像剪切与频域变换示例
- `005.py` - 二维离散余弦变换与逆变换示例

## 相关技术

- 图像几何变换: 平移、缩放、旋转、转置
- 插值算法: 最近邻、双线性、双三次
- 频域分析: 傅里叶变换、离散余弦变换 
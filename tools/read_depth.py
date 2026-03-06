import numpy as np
import imageio.v2 as imageio

path = "data0/replica/room0/depth/depth000000.png"
img = imageio.imread(path)           # 读为 numpy 数组
print("shape:", img.shape, "dtype:", img.dtype)
print("min:", img.min(), "max:", img.max(), "mean:", float(img.mean()))
print("matrix:\n", img)              # 输出完整矩阵（可能很大）
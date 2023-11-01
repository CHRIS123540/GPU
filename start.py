import matplotlib.pyplot as plt
import numpy as np

# 生成随机线性数据
x = np.random.rand(50)
y = 2 * x + 1 + np.random.randn(50)

# 绘制图像
plt.scatter(x, y)
plt.plot(x, 2*x + 1, color='red')  # 红线是原始线性关系

# 保存图像到文件
plt.savefig('plot.png')

# 关闭图像（可选）
plt.close()

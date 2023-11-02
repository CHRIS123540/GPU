import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx

# 1. 训练一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 1)  # 一个简单的线性层

    def forward(self, x):
        return self.fc(x)

# 生成一些模拟数据
x_train = torch.randn(1000, 1) * 10
y_train = 3 * x_train + 5 + torch.randn(1000, 1) * 1  # y = 3x + 5 + noise

# 创建模型、损失函数和优化器
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(2000):
    model.train()
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')
#
# 2. 保存模型为 ONNX 格式
# 定义一个模拟输入
x = torch.randn(1, 1)
# 导出模型到 ONNX 格式
torch.onnx.export(model, x, "./model/model.onnx")

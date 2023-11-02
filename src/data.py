import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import os

def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def process_data(images_path, labels_path):
    data = load_mnist_images(images_path)
    labels = load_mnist_labels(labels_path)
    data = data.astype('float32') / 255
    labels = torch.tensor(labels, dtype=torch.long)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data_tensor, labels)
    return dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

train_dataset = process_data('/home/hlx/work/GPU/src/data/MNIST/raw/train-images-idx3-ubyte', '/home/hlx/work/GPU/src/data/MNIST/raw/train-labels-idx1-ubyte')
test_dataset = process_data('/home/hlx/work/GPU/src/data/MNIST/raw/t10k-images-idx3-ubyte', '/home/hlx/work/GPU/src/data/MNIST/raw/t10k-labels-idx1-ubyte')

train_dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = SimpleNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
for epoch in range(10):  # reduce epoch to 10 for quicker training
    model.train()
    for data, target in train_dataloader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Training Loss: {loss.item():.4f}')

# Testing the model after training
model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_dataloader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_dataloader.dataset)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {correct}/{len(test_dataloader.dataset)} ({100. * correct / len(test_dataloader.dataset):.0f}%)')
model.eval()

# 创建一个假的输入张量以表示模型的输入
dummy_input = torch.randn(1, 28, 28).to(device)  # 这里的形状应该是正确的

# 使用 torch.onnx 导出模型，并指定动态轴
torch.onnx.export(
    model,
    dummy_input,
    "mnist_model.onnx",
    input_names=["input"],		# 输入名
    output_names=["output"],	# 输出名
    dynamic_axes={
        'input': {0: 'batch_size'},  # 'input' 和 'output' 是你模型的输入/输出名称
        'output': {0: 'batch_size'}  # 如果你的模型有不同的名称，请更新这些
    }
)

print("Model saved as mnist_model.onnx")

# 删除临时模型文件（如果你创建了一个）
temp_model_path = "temp_model.onnx"
if os.path.exists(temp_model_path):
    os.remove(temp_model_path)
    print(f"Temporary model file {temp_model_path} deleted.")
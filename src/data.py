import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

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

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(6*6*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 6*6*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training the model
for epoch in range(50):
    model.train()
    for data, target in train_dataloader:
        data = data.view(-1, 1, 28, 28).to(device)
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
        data = data.view(-1, 1, 28, 28).to(device)
        target = target.to(device)
        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_dataloader.dataset)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {correct}/{len(test_dataloader.dataset)} ({100. * correct / len(test_dataloader.dataset):.0f}%)')

# ... (您之前的代码)

# 将模型转为评估模式
model.eval()

# 创建一个假的输入张量以表示模型的输入
dummy_input = torch.randn(1, 1, 28, 28).to(device)

# 使用 torch.onnx 导出模型
torch.onnx.export(model, dummy_input, "mnist_model.onnx")

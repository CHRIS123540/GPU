import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class Config(object):
    def __init__(self, class_list):
        self.model_name = 'TSCRNN'
        self.class_list = class_list
        self.save_path = './saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = 0.001
        self.num_classes = len(self.class_list)
        self.num_epochs = 30
        self.batch_size = 128

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=15, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64, affine=True)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64, affine=True)
        self.lstm = nn.LSTM(375, 256, 2, bidirectional=True, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(in_features=512, out_features=config.num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = F.max_pool1d(out, kernel_size=2, stride=2)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = F.max_pool1d(out, kernel_size=2, stride=2)

        out, _ = self.lstm(out)
        out = self.dropout(out)

        out = self.out(out[:, -1, :])
        return out
# 创建随机输入
def generate_random_input(batch_size, sequence_length, feature_dim):
    return torch.randn(batch_size, sequence_length, feature_dim)

# 示例使用
if __name__ == "__main__":
    # 定义类别列表
    class_labels = ['class1', 'class2']
    
    # 初始化配置
    config = Config(class_labels)
    
    # 初始化模型
    model = Model(config)
    
    # 随机生成输入
    batch_size = 1
    sequence_length = 1500  # 序列长度
    feature_dim = 15       # 特征维度
    random_input = generate_random_input(batch_size,feature_dim, sequence_length )
    
    # 将模型设置为评估模式
    model.eval()
    
    # 运行模型
    with torch.no_grad():
        output = model(random_input)
    
    # 打印输出
    print("模型输出:", output)

    import time

    # 测试设备性能的函数
    def test_device_performance(device, model, test_input):
        model.eval()
        total_time = 0.0
        num_runs = 2000  # 运行多次取平均值
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                model(test_input)
            end_time = time.time()
            total_time += end_time - start_time
        average_time = total_time / num_runs
        return average_time

    # 测试CPU性能
    model_cpu = Model(config).to('cpu')
    test_input_cpu = torch.randn(1, 15, 1500)
    average_time_cpu = test_device_performance(torch.device('cpu'), model_cpu, test_input_cpu)
    print(f"Average forward pass time on CPU: {average_time_cpu:.6f} seconds")

    # 如果可用，测试GPU性能
    if torch.cuda.is_available():
        device_gpu = torch.device("cuda")
        model_gpu = Model(config).to(device_gpu)
        test_input_gpu = test_input_cpu.to(device_gpu)
        average_time_gpu = test_device_performance(device_gpu, model_gpu, test_input_gpu)
        print(f"Average forward pass time on GPU: {average_time_gpu:.6f} seconds")
    else:
        print("GPU is not available.")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class Config(object):
    def __init__(self, class_list):
        self.model_name = 'datanet'
        self.class_list = class_list
        self.save_path = './saved_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = 0.01
        self.num_classes = len(self.class_list)
        self.num_epochs = 100
        self.batch_size = 128

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features=1480, out_features=128) 
        self.fc2 = nn.Linear(in_features=128, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=config.num_classes)
        
    def forward(self, x):        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.out(x)
        return x

# 创建随机输入
def generate_random_input(batch_size, feature_dim):
    return torch.randn(batch_size, feature_dim)

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

# 示例使用
if __name__ == "__main__":
    class_labels = ['class1', 'class2']

    # 初始化配置
    config = Config(class_labels)

    # 初始化模型
    model = Model(config)

    # 随机生成输入
    batch_size = 1
    feature_dim = 1480  # 特征维度
    random_input = generate_random_input(batch_size, feature_dim)

    # 将模型设置为评估模式
    model.eval()

    # 运行模型
    with torch.no_grad():
        output = model(random_input)

    print("模型输出:", output)

    # 测试CPU性能
    model_cpu = Model(config).to('cpu')
    test_input_cpu = torch.randn(1, 1480)
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

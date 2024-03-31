import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# 配置类
class Config(object):
    def __init__(self):
        self.hidden_size = 100
        self.num_classes = 10  # 假设有10个类别
        self.dropout = 0.1

# 模型类
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(1500, config.hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        # 在初始化时不指定设备
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.fc1 = nn.Linear(config.hidden_size * 2, 128)
        self.fc2 = nn.Linear(128, config.num_classes)

    def forward(self, x):
        H, _ = self.lstm(x)
        # 确保参数在相同设备
        w_device = H.device
        self.w = self.w.to(w_device)
        alpha = F.softmax(torch.matmul(H, self.w), dim=1).unsqueeze(-1)
        out = H * alpha
        out = torch.sum(out, 1)
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def test_device_performance(device, model, test_input):
    torch.cuda.synchronize(device) if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            output = model(test_input)
    torch.cuda.synchronize(device) if device.type == 'cuda' else None
    end_time = time.time()
    return (end_time - start_time) / num_runs

config = Config()

# 测试CPU性能
model = Model(config).to('cpu')
test_input_cpu = torch.randn(1, 10, 1500)
num_runs = 2000
average_time_cpu = test_device_performance(torch.device('cpu'), model, test_input_cpu)
print(f"Average forward pass time on CPU: {average_time_cpu:.6f} seconds")

# 如果可用，测试GPU性能
if torch.cuda.is_available():
    device_gpu = torch.device("cuda")
    model = Model(config).to(device_gpu)
    test_input_gpu = test_input_cpu.to(device_gpu)
    average_time_gpu = test_device_performance(device_gpu, model, test_input_gpu)
    print(f"Average forward pass time on GPU: {average_time_gpu:.6f} seconds")
else:
    print("GPU is not available.")


# 定义一个文件名用于保存ONNX模型
onnx_file_path = 'BiLSTM.onnx'
x = torch.randn(1, 10, 1500, device=device_gpu)
# 导出模型
# 这里假设模型已经被移到了正确的设备上
torch.onnx.export(model,               # 运行模型的实例
                  x,                   # 模型的输入，可以是一个或多个张量
                  onnx_file_path,      # 保存ONNX模型的文件路径
                  export_params=True,  # 是否导出模型参数
                  opset_version=11,    # ONNX操作集的版本
                  do_constant_folding=True,  # 是否执行常量折叠优化
                  input_names=['input'],    # 输入张量的名字
                  output_names=['output'],  # 输出张量的名字
                  dynamic_axes={'input': {0: 'batch_size'},  # 批量大小动态化
                                'output': {0: 'batch_size'}})

print(f"Model was successfully exported to {onnx_file_path}")
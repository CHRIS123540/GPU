
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import time


class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'MATEC'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

        self.dropout = 0.1                                             
        self.require_improvement = 10000                               
        self.num_classes = 1                       
        self.n_vocab = 0                                               
        self.num_epochs = 50                                         
        self.batch_size = 128                                          
        self.pad_size = 786                                              
        self.learning_rate = 0.0005                                      
        self.embed =  432          
        self.dim_model = 432
        self.num_head = 3
        self.num_encoder = 2


'''Attention Is All You Need'''
class Scaled_Dot_Product_Attention(nn.Module):
    '''Scaled Dot-Product Attention '''
    def __init__(self):
        super(Scaled_Dot_Product_Attention, self).__init__()

    def forward(self, Q, K, V, scale=None):
        attention = torch.matmul(Q, K.permute(0, 2, 1))
        if scale:
            attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    def __init__(self, dim_model, num_head, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_head = num_head
        assert dim_model % num_head == 0
        self.dim_head = dim_model // self.num_head
        self.fc_Q = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_K = nn.Linear(dim_model, num_head * self.dim_head)
        self.fc_V = nn.Linear(dim_model, num_head * self.dim_head)
        self.attention = Scaled_Dot_Product_Attention()
        self.fc = nn.Linear(num_head * self.dim_head, dim_model)
       

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        scale = K.size(-1) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        return out

class Add_Norm(nn.Module):
    def __init__(self, dim_model):
        super(Add_Norm, self).__init__()
        self.fc1 = nn.Linear(dim_model, dim_model)
        self.layer_norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(0.1)
    def forward(self, origin_x, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = out + origin_x  
        out = self.dropout(out)
        out = self.layer_norm(out)
        return out 
    

    
class Feed_Forward(nn.Module):
    def __init__(self, dim_model):
        super(Feed_Forward, self).__init__()
        self.conv = nn.Conv1d(in_channels=3, out_channels=3, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.conv(x)
        return out
    
class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Feed_Forward(dim_model)
        self.add_norm1 = Add_Norm(dim_model)
        self.add_norm2 = Add_Norm(dim_model)
        
    def forward(self, x):
        
        out = self.attention(x)
        out = self.add_norm1(x,out)
        out_ = self.feed_forward(out)
        out = self.add_norm2(out,out_)
        return out


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        
        self.fc1 = nn.Linear(config.pad_size,config.dim_model)
        self.encoder1 = Encoder(config.dim_model, config.num_head, config.dropout)
        self.encoder2 = Encoder(config.dim_model, config.num_head, config.dropout)
        # self.fc2 = nn.Linear(config.last_hidden, config.num_classes)
        self.fc2 = nn.Linear(1296, config.num_classes)
        self.dropout = nn.Dropout(0.1)
    def forward(self, x):
        
        out = self.fc1(x)
        out = self.encoder1(out)
        out = self.encoder2(out)
        out = torch.flatten(out,start_dim=1)
        out = self.dropout(out)
        out = self.fc2(out)
    
        return out
def test_device_performance(device, model, test_input, num_runs=2000):
    """测试指定设备上模型的性能"""
    # 确保模型在正确的设备上
    model.to(device)
    test_input = test_input.to(device)
    
    # 同步CUDA（如果使用）
    torch.cuda.synchronize(device) if device.type == 'cuda' else None
    
    start_time = time.time()
    
    # 执行多次前向传播以计算平均时间
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(test_input)
    
    # 再次同步以确保所有计算完成
    torch.cuda.synchronize(device) if device.type == 'cuda' else None
    
    end_time = time.time()
    
    # 计算并返回平均执行时间
    avg_time = (end_time - start_time) / num_runs
    return avg_time

if __name__ == '__main__':
    config = Config()

    # 创建模型输入
    x1 = torch.rand(10, 1, 786)  # 输入的形状调整为符合模型期望的形状
    x2 = torch.rand(10, 1, 786)
    x3 = torch.rand(10, 1, 786)
    test_input = torch.cat((x1, x2, x3), dim=1)  # 这里假设每次只处理一个批次的数据，每个批次包含3个样本

    # 测试CPU性能
    model_cpu = Model(config).to('cpu')
    average_time_cpu = test_device_performance(torch.device('cpu'), model_cpu, test_input, num_runs=2000)
    print(f"Average forward pass time on CPU: {average_time_cpu:.6f} seconds")

    # 如果可用，测试GPU性能
    if torch.cuda.is_available():
        model_gpu = Model(config).to('cuda')
        average_time_gpu = test_device_performance(torch.device('cuda'), model_gpu, test_input, num_runs=2000)
        print(f"Average forward pass time on GPU: {average_time_gpu:.6f} seconds")
    else:
        print("GPU is not available.")

    # 定义一个文件名用于保存ONNX模型
    onnx_file_path = 'MATEC.onnx'
    batch_size = 10
    seq_len = 3
    feature_size = 786

    # 构造一个代表性的输入张量
    dummy_input = torch.randn(batch_size, seq_len, feature_size).to(config.device)

    # 设置ONNX模型保存的路径

    # 导出模型
    torch.onnx.export(model_gpu,                     # 模型实例
                    dummy_input,               # 模型的输入占位符
                    onnx_file_path,            # ONNX模型保存路径
                    export_params=True,        # 导出模型参数
                    opset_version=11,          # 设置ONNX版本
                    do_constant_folding=True,  # 是否执行常量折叠优化
                    input_names=['input'],     # 输入名
                    output_names=['output'],   # 输出名
                    dynamic_axes={'input': {0: 'batch_size'},  # 使批量大小动态
                                    'output': {0: 'batch_size'}})

    print(f"Model has been converted to ONNX: {onnx_file_path}")
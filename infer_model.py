import numpy as np
import torch
import onnxruntime
import time  # 导入time模块

# 加载 ONNX 模型
ort_session = onnxruntime.InferenceSession("./model/model.onnx")

# 指定使用GPU 2进行推理
ort_session.set_providers(['CUDAExecutionProvider'], [{'device_id': 0}])

def infer_model(x_test):
    # 定义输入数据
    ort_inputs = {ort_session.get_inputs()[0].name: x_test}
    
    # 在推理开始前暂停5秒，这时您可以运行nvidia-smi
    time.sleep(5)
    
    # 进行推理
    ort_outs = ort_session.run(None, ort_inputs)
    
    # 在推理结束后再暂停5秒，这时您还可以再次运行nvidia-smi检查
    time.sleep(2)
    
    return ort_outs[0]

x_test = np.array([[1.0]], dtype=np.float32)
result = infer_model(x_test)
print(result)

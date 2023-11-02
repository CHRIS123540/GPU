
import onnxruntime as ort
import numpy as np
from PIL import Image

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# 加载图像
image_path = "4.png"  # 替换为您的图像路径
image = Image.open(image_path).convert('L')  # 转为灰度图

# 对图像进行处理
image = np.asarray(image, dtype=np.float32) / 255.0
image = image.reshape(1, 1, 28, 28)

# 加载 ONNX 模型

# 创建 ONNX 运行时会话
ort_session = ort.InferenceSession("mnist_model.onnx")

# 进行推断
ort_inputs = {ort_session.get_inputs()[0].name: image}
ort_outs = ort_session.run(None, ort_inputs)
predicted_class = np.argmax(ort_outs[0])

print(f"Predicted Class: {predicted_class}")

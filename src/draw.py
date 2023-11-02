import onnxruntime as ort
import numpy as np
from PIL import Image

def load_and_process_image(image_path):
    image = Image.open(image_path).convert('L')  # 转为灰度图
    image = np.asarray(image, dtype=np.float32) / 255.0
    return image

# 加载图像
image_paths = ["1.png", "3.png", "5.png"]  # 替换为您的图像路径
images = [load_and_process_image(image_path) for image_path in image_paths]

# 将图像堆叠成一个大的NumPy数组
input_array = np.stack(images)

# 确保输入数组的形状是 (num_images, 28, 28)
assert input_array.shape == (len(image_paths), 28, 28)

# 加载 ONNX 模型
ort_session = ort.InferenceSession("mnist_model.onnx")

# 进行推断
ort_inputs = {ort_session.get_inputs()[0].name: input_array}
ort_outs = ort_session.run(None, ort_inputs)
predicted_classes = np.argmax(ort_outs[0], axis=1)

# 输出每个图像的预测类别
for image_path, predicted_class in zip(image_paths, predicted_classes):
    print(f"Predicted Class for {image_path}: {predicted_class}")

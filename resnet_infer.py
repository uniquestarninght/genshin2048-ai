# resnet_infer.py
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np


class ResNetPredictor:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 尝试加载模型 - 兼容新旧格式
        checkpoint = torch.load(model_path, map_location=self.device)

        # 如果checkpoint是字典格式（包含类别信息和其他信息）
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                # 新格式：包含类别信息
                state_dict = checkpoint['model_state_dict']
                if 'class_names' in checkpoint:
                    self.class_names = [int(name) for name in checkpoint['class_names']]
                    self.num_classes = checkpoint['num_classes']
                else:
                    # 如果没有类别信息，使用默认的
                    ALL_CLASS_VALUES = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
                    self.class_names = ALL_CLASS_VALUES
                    self.num_classes = len(ALL_CLASS_VALUES)
            else:
                # 如果是旧格式，直接使用checkpoint作为state_dict
                state_dict = checkpoint
                # 尝试从模型文件名推断类别数量（不推荐，但为了兼容性）
                ALL_CLASS_VALUES = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
                self.class_names = ALL_CLASS_VALUES
                self.num_classes = len(ALL_CLASS_VALUES)
        else:
            # 旧格式：直接是state_dict
            state_dict = checkpoint
            ALL_CLASS_VALUES = [0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
            self.class_names = ALL_CLASS_VALUES
            self.num_classes = len(ALL_CLASS_VALUES)

        # 构建模型结构
        self.model = models.resnet34(weights=None)  # 不加载预训练权重
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)

        # 加载训练好的权重
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

        print(f"模型加载成功 - 类别数量: {self.num_classes}, 类别: {self.class_names}")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((184, 184)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, cv2_image):
        """
        输入: BGR numpy array (H, W, C)
        输出: (predicted_number, confidence)
        """
        try:
            if cv2_image is None or cv2_image.size == 0:
                return 0, 0.0

            # 转 RGB
            rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            tensor = self.transform(rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor)
                probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probabilities)
                confidence = float(probabilities[pred_idx])
                predicted_number = self.class_names[pred_idx]  # 使用动态类别名称

            return predicted_number, confidence

        except Exception as e:
            # 如果出错，返回默认值
            print(f"ResNet 预测出错: {e}")
            return 0, 0.0

    def get_class_names(self):
        """返回模型能够识别的类别"""
        return self.class_names

    def get_num_classes(self):
        """返回类别数量"""
        return self.num_classes

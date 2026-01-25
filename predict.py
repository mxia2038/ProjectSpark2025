import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import torch.optim as optim


# 确保模型和配置已经加载
class Config:
    model_save_path = "C:/Users/Admin/PycharmProjects/2024/output/crystal_stage_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 6  # 6个阶段


# 加载模型
model = models.resnet34(pretrained=True)  # 使用预训练的 ResNet34
model.fc = nn.Sequential(
    nn.BatchNorm1d(model.fc.in_features),
    nn.Dropout(0.3),
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.3),
    nn.Linear(512, Config.num_classes)
)

# 加载训练好的模型权重
model.load_state_dict(torch.load(Config.model_save_path))
model = model.to(Config.device)
model.eval()  # 切换到评估模式


# 图像预处理
def process_image(image_path):
    image = Image.open(image_path).convert('RGB')  # 强制转换为 RGB 图像
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image).unsqueeze(0)  # 增加 batch 维度
    return image


# 预测函数
def predict(image_path):
    image = process_image(image_path)
    image = image.to(Config.device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()  # 获取预测类别
    print(f"Predicted Stage: {predicted_class}")


# 测试一张新的图像
image_path = 'C:/Users/Admin/PycharmProjects/2024/test_sample4.jpg'  # 替换为图像路径
predict(image_path)

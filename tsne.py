import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np

# 加载预训练的 ResNet34 模型
model = models.resnet34(pretrained=True)

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 修改模型以提取特征
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])  # 去掉最后的全连接层

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 展平特征
        return x

# 创建特征提取器
feature_extractor = FeatureExtractor(model).eval()

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # 减少翻转概率
    transforms.RandomRotation(15),  # 减小旋转角度
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 添加颜色扰动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载数据集
dataset = datasets.ImageFolder(root="C:/Users/Admin/PycharmProjects/2024/dataset/train", transform=transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

features = []
labels = []

with torch.no_grad():
    for images, targets in dataloader:
        outputs = feature_extractor(images)
        features.append(outputs.cpu().numpy())
        labels.extend(targets.cpu().numpy())

# 将特征和标签转换为 NumPy 数组
features = np.concatenate(features)
labels = np.array(labels)


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42)
tsne_features = tsne.fit_transform(features)

# 绘制 t-SNE 可视化结果
plt.figure(figsize=(10, 8))
scatter = plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels, cmap='viridis', s=10)
plt.colorbar(scatter, label='Class labels')
plt.title("t-SNE Visualization of ResNet34 Features")
plt.xlabel("t-SNE Feature 1")
plt.ylabel("t-SNE Feature 2")
plt.savefig('tsne.png')

import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import torch
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch import serialization

# # 设置 scienceplots 的学术风格
# plt.style.use(['science', 'ieee'])

# --------------------- 配置参数 ---------------------
class Config:
    model_save_path = "C:/Users/Admin/PycharmProjects/2024/output/crystal_stage_model.pth"
    results_save_path = "C:/Users/Admin/PycharmProjects/2024/output/training_results.npz"
    test_loader_path = "C:/Users/Admin/PycharmProjects/2024/output/test_loader.pth"
    num_classes = 6  # 6个阶段
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------- 加载训练结果 ---------------------
results = np.load(Config.results_save_path)
train_losses = results['train_losses']
val_losses = results['val_losses']
train_accuracies = results['train_accuracies']
val_accuracies = results['val_accuracies']
train_precisions = results['train_precisions']
val_precisions = results['val_precisions']
train_recalls = results['train_recalls']
val_recalls = results['val_recalls']

# --------------------- 绘图函数 ---------------------
def plot_results():
    epochs = range(1, len(train_losses) + 1)

    # 图像 1: 损失曲线
    # with plt.style.context(['science', 'ieee']):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300)
    plt.close()

    # 图像 2: 准确率曲线
    # with plt.style.context(['science', 'ieee']):
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_accuracies, label='Train Accuracy', linewidth=2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', linewidth=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Accuracy (\%)', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('accuracy_curve.png', dpi=300)
    plt.close()
#
#     # 图像 3: Precision 曲线
#     with plt.style.context(['science', 'ieee']):
#         plt.figure(figsize=(8, 6))
#         plt.plot(epochs, train_precisions, label='Train Precision', linewidth=2)
#         plt.plot(epochs, val_precisions, label='Validation Precision', linewidth=2)
#         plt.xlabel('Epochs', fontsize=12)
#         plt.ylabel('Precision', fontsize=12)
#         plt.legend(fontsize=12)
#         plt.tight_layout()
#         plt.savefig('precision_curve.png', dpi=300)
#         plt.close()
#
#     # 图像 4: Recall 曲线
#     with plt.style.context(['science', 'ieee']):
#         plt.figure(figsize=(8, 6))
#         plt.plot(epochs, train_recalls, label='Train Recall', linewidth=2)
#         plt.plot(epochs, val_recalls, label='Validation Recall', linewidth=2)
#         plt.xlabel('Epochs', fontsize=12)
#         plt.ylabel('Recall', fontsize=12)
#         plt.legend(fontsize=12)
#         plt.tight_layout()
#         plt.savefig('recall_curve.png', dpi=300)
#         plt.close()
#
#     print("Plots saved successfully!")


# # --------------------- 混淆矩阵 ---------------------
# def plot_confusion_matrix():
#     # 加载模型
#     model = models.resnet34(pretrained=False)
#     model.fc = nn.Sequential(
#         nn.BatchNorm1d(model.fc.in_features),
#         nn.Dropout(0.3),
#         nn.Linear(model.fc.in_features, 512),
#         nn.ReLU(),
#         nn.BatchNorm1d(512),
#         nn.Dropout(0.3),
#         nn.Linear(512, Config.num_classes)
#     )
#     model.load_state_dict(torch.load(Config.model_save_path))
#     model = model.to(Config.device)
#
#     from torch.utils.data import Subset
#     torch.serialization.add_safe_globals([Subset])
#     test_loader = torch.load(Config.test_loader_path, weights_only=False)
#
#     model.eval()  # 设置模型为评估模式
#     all_labels = []
#     all_preds = []
#
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs, labels = inputs.to(Config.device), labels.to(Config.device)
#             outputs = model(inputs)
#             _, predicted = torch.max(outputs, 1)
#             all_labels.extend(labels.cpu().numpy())
#             all_preds.extend(predicted.cpu().numpy())
#
#     cm = confusion_matrix(all_labels, all_preds)
#
#     # 使用 seaborn 绘制热力图
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2, 3], yticklabels=[0, 1, 2, 3])
#     plt.xlabel('Predicted', font={'family':'Times New Roman', 'size':12})
#     plt.ylabel('True', font={'family':'Times New Roman', 'size':12})
#     plt.tight_layout()
#     plt.savefig('confusion_matrix.png', dpi=300)
#     plt.close()
#
#     print("Confusion matrix plot saved as confusion_matrix.png")


# --------------------- 主程序 ---------------------
if __name__ == "__main__":
    # 绘制图表和混淆矩阵
    plot_results()
    #plot_confusion_matrix()


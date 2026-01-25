import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import os
from mymodel import SimpleCNN1

# --------------------- 配置参数 ---------------------
class Config:
    data_dir = "C:/Users/Admin/PycharmProjects/2024/dataset/train"  # 数据集路径
    batch_size = 16
    num_epochs = 30
    lr = 3e-4
    num_classes = 4  # 4个阶段
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = "C:/Users/Admin/PycharmProjects/2024/output/crystal_stage_model.pth"
    results_save_path = "C:/Users/Admin/PycharmProjects/2024/output/training_results.npz"  # 保存训练结果
    test_loader_path = "C:/Users/Admin/PycharmProjects/2024/output/test_loader.pth"  # 保存测试集数据加载器

# 全局变量用于记录训练结果
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_precisions = []
val_precisions = []
train_recalls = []
val_recalls = []

# --------------------- 数据预处理和加载 ---------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),  # 减少翻转概率
    transforms.RandomRotation(15),  # 减小旋转角度
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 添加颜色扰动
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

import tqdm
# --------------------- 训练和验证函数 ---------------------
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    for inputs, labels in tqdm.tqdm(train_loader):
        inputs, labels = inputs.to(Config.device), labels.to(Config.device)

        optimizer.zero_grad()  # 清除旧的梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # 收集所有标签和预测结果
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    # 计算平均损失和准确率
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    # 计算 Precision 和 Recall
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    # 记录训练集损失、准确率、Precision 和 Recall
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)
    train_precisions.append(precision)
    train_recalls.append(recall)

    # 输出训练结果
    print(f"Epoch [{epoch + 1}/{Config.num_epochs}], "
          f"Train Loss: {avg_loss:.4f}, "
          f"Train Accuracy: {accuracy:.2f}%, "
          f"Train Precision: {precision:.4f}, "
          f"Train Recall: {recall:.4f}")
    return accuracy


def validate(model, val_loader, criterion):
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)

            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # 收集所有标签和预测结果
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 计算平均损失和准确率
    avg_loss = running_loss / len(val_loader)
    accuracy = 100 * correct / total

    # 计算 Precision 和 Recall
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

    # 记录验证集损失、准确率、Precision 和 Recall
    val_losses.append(avg_loss)
    val_accuracies.append(accuracy)
    val_precisions.append(precision)
    val_recalls.append(recall)

    # 输出验证结果
    print(f"Validation Loss: {avg_loss:.4f}, "
          f"Validation Accuracy: {accuracy:.2f}%, "
          f"Validation Precision: {precision:.4f}, "
          f"Validation Recall: {recall:.4f}")
    return accuracy


# --------------------- 混淆矩阵 ---------------------
def plot_confusion_matrix(model, test_loader):
    model.eval()  # 设置模型为评估模式
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 手动设置标签范围（0 到 5）
    labels_range = range(Config.num_classes)
    cm = confusion_matrix(all_labels, all_preds, labels=labels_range)

    # 使用 seaborn 绘制热力图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_range, yticklabels=labels_range)
    plt.title('Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.close()

    print("Confusion matrix plot saved as confusion_matrix.png")


# --------------------- 训练主程序 ---------------------
if __name__ == "__main__":
    # 加载数据
    full_dataset = datasets.ImageFolder(Config.data_dir, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)

    # 定义模型
    model = models.resnet34(pretrained=True)
    model.fc = nn.Sequential(
        nn.BatchNorm1d(model.fc.in_features),
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, Config.num_classes)
    )

    model = model.to(Config.device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=Config.num_epochs
    )
    # 训练模型
    best_val_accuracy = 0.0
    for epoch in range(Config.num_epochs):
        train_accuracy = train(model, train_loader, criterion, optimizer, epoch)
        val_accuracy = validate(model, val_loader, criterion)

        # Step the scheduler
        scheduler.step()

        # 保存最佳模型
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), Config.model_save_path)

    # 保存训练结果
    np.savez(
        Config.results_save_path,
        train_losses=train_losses,
        val_losses=val_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        train_precisions=train_precisions,
        val_precisions=val_precisions,
        train_recalls=train_recalls,
        val_recalls=val_recalls,
    )
    print(f"Training results saved to {Config.results_save_path}")

    # 保存测试集数据加载器
    torch.save(test_loader, Config.test_loader_path)
    print(f"Test loader saved to {Config.test_loader_path}")

    # 绘制混淆矩阵
    plot_confusion_matrix(model, test_loader)
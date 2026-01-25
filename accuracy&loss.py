import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # 导入 Science Plots 库

# 应用 Science Plots 的 IEEE 样式（适合学术论文）
plt.style.use(['science', 'ieee'])  # 使用 IEEE 样式

# 读取 CSV 文件（假设文件名为 'data.csv'）
# CSV 文件应包含以下列：epoch, Train Accuracy, Validation Accuracy, Train Loss, Validation Loss
data = pd.read_csv('C:/Users/Admin/PycharmProjects/2024/dataset/dldata.csv')

# 提取数据
epoch = data['Epoch']
train_acc = data['Train Accu']
val_acc = data['Val Accu']
train_acc = [acc * 100 for acc in train_acc]
val_acc = [acc * 100 for acc in val_acc]
train_loss = data['Train Loss']
val_loss = data['Val Loss']
train_precisions= data['Train precisions']
val_precisions = data['Val precisions']
train_recalls= data['Train recalls']
val_recalls = data['Val recalls']

# 创建画布和双 Y 轴
fig, ax1 = plt.subplots(figsize=(8, 6))

# 左侧 Y 轴：Accuracy
ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('Accuracy', fontsize=14)
ax1.plot(epoch, train_acc, marker='', linestyle='-', linewidth=2, color='blue', label='Train Accuracy')
ax1.plot(epoch, val_acc, marker='', linestyle='--', linewidth=2, color='orange', label='Validation Accuracy')
ax1.tick_params(axis='y', labelsize=14)
ax1.set_ylim(0, 105)  # 设置 Accuracy 范围为 0 到 1

# 右侧 Y 轴：Loss
ax2 = ax1.twinx()
ax2.set_ylabel('Loss', fontsize=14)
ax2.plot(epoch, train_loss, marker='', linestyle=':', linewidth=2, color='firebrick', label='Train Loss')
ax2.plot(epoch, val_loss, marker='', linestyle='-.', linewidth=2, color='green', label='Validation Loss')
ax2.tick_params(axis='y', labelsize=14)
ax2.set_ylim(0, max(max(train_loss), max(val_loss)) * 1.1)  # 设置 Loss 范围为 0 到最大 Loss 的 1.1 倍

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', bbox_to_anchor=(1.0, 0.5),fontsize=12)

# 保存为高分辨率图片
plt.tight_layout()
plt.savefig('accuracy_loss_plot.png', dpi=600, bbox_inches='tight')
plt.show()

# 图像 3: Precision 曲线
plt.figure(figsize=(8, 6))
plt.plot(epoch, train_precisions, label='Train Precision', linewidth=2)
plt.plot(epoch, val_precisions, label='Validation Precision', color='firebrick',linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('precision_curve.png', dpi=600)
plt.show()

# 图像 4: Recall 曲线
plt.figure(figsize=(8, 6))
plt.plot(epoch, train_recalls, label='Train Recall', linewidth=2)
plt.plot(epoch, val_recalls, label='Validation Recall', color='firebrick',linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Recall', fontsize=12)
plt.legend(loc = 'lower right',fontsize=12)
plt.tight_layout()
plt.savefig('recall_curve.png', dpi=300)
plt.show()
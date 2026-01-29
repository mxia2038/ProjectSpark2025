# 晶体生长阶段分类

基于 ResNet34 的晶体生长阶段深度学习分类模型。

## 项目概述

本项目实现了一个 6 类图像分类模型，用于识别晶体生长的不同阶段（0-5）。模型采用 ImageNet 预训练的 ResNet34 进行迁移学习。

## 功能特性

- 基于 ResNet34 的分类模型，自定义全连接层
- 数据增强（随机翻转、旋转、颜色抖动）
- 训练可视化（损失曲线、准确率指标）
- t-SNE 特征降维可视化
- 混淆矩阵分析
- **GUI 桌面应用** - 基于 PyQt5 的桌面应用：
  - 中文界面（反应自动控制系统）
  - 图像选择与模拟拍照
  - 实时阶段预测与置信度显示
  - 6 个阶段的概率可视化
  - 检测历史记录
  - 手动修正与数据收集
  - 高 DPI 显示支持

## 环境配置

```bash
pip install -r requirements.txt
```

## 项目结构

```
├── main.py              # 训练主程序（含模型架构定义）
├── predict.py           # 单张图像预测
├── crystal_classifier.py # 模型封装类
├── gui_app.py           # PyQt5 GUI 应用
├── confusionmatrix.py   # 混淆矩阵可视化
├── tsne.py              # t-SNE 特征降维可视化
├── accuracy&loss.py     # 训练指标绘图
├── plot_results.py      # 结果可视化
├── var_pred.py          # 多元线性回归分析
└── requirements.txt     # 依赖列表
```

## 使用方法

### 模型训练

```python
python main.py
```

训练配置（在 `main.py` 中）：
- `batch_size`: 16
- `num_epochs`: 30
- `lr`: 3e-4
- `num_classes`: 6

### 单张预测

```python
python predict.py
```

### GUI 应用

```python
python gui_app.py
```

功能说明：
- **选择图像**：选择本地图像文件进行预测
- **模拟拍照**：从数据集中随机选取图像
- **开始自动模式**：连续检测模式（每 2.5 秒）
- **手动修正**：将误分类图像保存到正确的文件夹，用于后续重新训练

## 模型架构

- **主干网络**：ResNet34（ImageNet 预训练）
- **输入尺寸**：224 × 224
- **输出**：6 类（阶段 0-5）
- **优化器**：AdamW（weight_decay=1e-4）
- **学习率调度**：OneCycleLR
- **损失函数**：CrossEntropyLoss（label_smoothing=0.1）

## 数据集

请按以下结构组织数据集：

```
dataset/
└── train/
    ├── 0/    # 阶段 0 图像
    ├── 1/    # 阶段 1 图像
    ├── 2/    # 阶段 2 图像
    ├── 3/    # 阶段 3 图像
    ├── 4/    # 阶段 4 图像
    └── 5/    # 阶段 5 图像
```

## 许可证

MIT License

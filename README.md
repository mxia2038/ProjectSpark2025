# Crystal Stage Classification

A deep learning model for classifying crystal growth stages using ResNet34.

## Overview

This project implements a 6-class image classification model to identify different stages of crystal formation. The model uses transfer learning with a pre-trained ResNet34 backbone.

## Features

- ResNet34-based classification with custom fully connected layers
- Data augmentation (random flip, rotation, color jitter)
- Training visualization (loss curves, accuracy metrics)
- t-SNE feature visualization
- Confusion matrix analysis
- **GUI Application** - PyQt5 desktop application with:
  - Chinese localized interface (反应自动控制系统)
  - Image selection and simulated camera capture
  - Real-time stage prediction with confidence display
  - Probability visualization for all 6 stages
  - Detection history tracking
  - Manual correction and data collection
  - High DPI display support

## Requirements

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── main.py              # Training script
├── mymodel.py           # Model architecture
├── predict.py           # Inference script
├── crystal_classifier.py # Model wrapper class
├── gui_app.py           # PyQt5 GUI application
├── confusionmatrix.py   # Confusion matrix visualization
├── tsne.py              # t-SNE feature visualization
├── accuracy&loss.py     # Training metrics plotting
├── plot_results.py      # Results visualization
├── var_pred.py          # Variable prediction analysis
└── requirements.txt     # Dependencies
```

## Usage

### Training

```python
python main.py
```

Configuration (in `main.py`):
- `batch_size`: 16
- `num_epochs`: 30
- `lr`: 3e-4
- `num_classes`: 6

### Prediction

```python
python predict.py
```

### GUI Application

```python
python gui_app.py
```

Features:
- **选择图像**: Select an image file for prediction
- **模拟拍照**: Randomly select an image from the dataset
- **开始自动模式**: Continuous detection mode (every 2.5 seconds)
- **手动修正**: Save misclassified images to correct folders for retraining

## Model Architecture

- **Backbone**: ResNet34 (ImageNet pre-trained)
- **Input size**: 224 x 224
- **Output**: 6 classes (stages 0-5)
- **Optimizer**: AdamW
- **Scheduler**: OneCycleLR
- **Loss**: CrossEntropyLoss with label smoothing

## Dataset

Prepare your dataset in the following structure:

```
dataset/
└── train/
    ├── 0/    # Stage 0 images
    ├── 1/    # Stage 1 images
    ├── 2/    # Stage 2 images
    ├── 3/    # Stage 3 images
    ├── 4/    # Stage 4 images
    └── 5/    # Stage 5 images
```

## License

MIT License

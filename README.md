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

## Requirements

```bash
pip install -r requirements.txt
```

## Project Structure

```
├── main.py              # Training script
├── mymodel.py           # Model architecture
├── predict.py           # Inference script
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

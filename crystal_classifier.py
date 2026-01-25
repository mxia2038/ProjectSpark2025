"""
Crystal Stage Classifier - Model wrapper for crystal stage classification.
Provides a clean interface for loading the model and making predictions.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from typing import Union, Tuple, List


class CrystalClassifier:
    """
    Crystal stage classifier using ResNet34.

    Usage:
        classifier = CrystalClassifier()
        stage, confidence, probs = classifier.predict("image.jpg")
    """

    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize the classifier.

        Args:
            model_path: Path to the model weights. If None, auto-detect.
            device: 'cuda' or 'cpu'. If None, auto-detect.
        """
        self.num_classes = 6
        self.stage_names = [f"Stage {i}" for i in range(self.num_classes)]

        # Auto-detect device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Auto-detect model path
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "output", "crystal_stage_model.pth")

        self.model_path = model_path

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # TTA (Test Time Augmentation) transforms
        self.tta_transforms = [
            self.transform,  # Original
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomRotation((90, 90)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
        ]

        # Load model
        self.model = self._build_model()
        self._load_weights()
        self.model.eval()

        print(f"Model loaded from: {model_path}")
        print(f"Device: {self.device}")

    def _build_model(self) -> nn.Module:
        """Build the ResNet34 model with custom classifier head."""
        model = models.resnet34(pretrained=False)
        model.fc = nn.Sequential(
            nn.BatchNorm1d(model.fc.in_features),
            nn.Dropout(0.3),
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, self.num_classes)
        )
        return model.to(self.device)

    def _load_weights(self):
        """Load model weights from file."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        state_dict = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

    def _preprocess(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocess an image for prediction.

        Args:
            image: Image path (str) or PIL Image

        Returns:
            Preprocessed tensor
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')
        else:
            raise ValueError("Image must be a file path or PIL Image")

        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image: Union[str, Image.Image]) -> Tuple[int, float, List[float]]:
        """
        Predict the crystal stage for an image.

        Args:
            image: Image path (str) or PIL Image

        Returns:
            Tuple of (predicted_stage, confidence, all_probabilities)
            - predicted_stage: int (0-5)
            - confidence: float (0-100, percentage)
            - all_probabilities: list of 6 floats (percentages for each stage)
        """
        tensor = self._preprocess(image)

        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

            predicted_stage = predicted.item()
            confidence_pct = confidence.item() * 100
            all_probs = [p * 100 for p in probabilities[0].cpu().tolist()]

        return predicted_stage, confidence_pct, all_probs

    def predict_with_tta(self, image: Union[str, Image.Image]) -> Tuple[int, float, List[float]]:
        """
        Predict with Test Time Augmentation (TTA) for more robust results.

        Args:
            image: Image path (str) or PIL Image

        Returns:
            Same as predict(), but averaged over augmented images
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            image = image.convert('RGB')

        all_probs = []

        with torch.no_grad():
            for transform in self.tta_transforms:
                tensor = transform(image).unsqueeze(0).to(self.device)
                outputs = self.model(tensor)
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs)

        # Average probabilities
        avg_probs = torch.mean(torch.stack(all_probs), dim=0)
        confidence, predicted = torch.max(avg_probs, 1)

        predicted_stage = predicted.item()
        confidence_pct = confidence.item() * 100
        all_probs_list = [p * 100 for p in avg_probs[0].cpu().tolist()]

        return predicted_stage, confidence_pct, all_probs_list

    def get_stage_name(self, stage: int) -> str:
        """Get the name for a stage number."""
        if 0 <= stage < self.num_classes:
            return self.stage_names[stage]
        return f"Unknown ({stage})"


# Simple test
if __name__ == "__main__":
    import sys

    classifier = CrystalClassifier()

    # Test with a sample image if provided
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Default test image
        base_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(base_dir, "test_sample.jpg")

    if os.path.exists(image_path):
        print(f"\nTesting with: {image_path}")

        stage, confidence, probs = classifier.predict(image_path)
        print(f"\nPrediction: {classifier.get_stage_name(stage)}")
        print(f"Confidence: {confidence:.1f}%")
        print("\nAll probabilities:")
        for i, prob in enumerate(probs):
            marker = " *" if i == stage else ""
            print(f"  Stage {i}: {prob:.1f}%{marker}")

        print("\n--- With TTA ---")
        stage_tta, conf_tta, probs_tta = classifier.predict_with_tta(image_path)
        print(f"Prediction: {classifier.get_stage_name(stage_tta)}")
        print(f"Confidence: {conf_tta:.1f}%")
    else:
        print(f"Test image not found: {image_path}")

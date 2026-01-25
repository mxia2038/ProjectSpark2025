"""
Crystal Stage Recognition System - GUI Application
A PyQt5-based desktop application for crystal stage classification.
"""

import os
import sys
import random
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QListWidget, QListWidgetItem,
    QProgressBar, QGroupBox, QFrame, QMessageBox, QSplitter, QComboBox
)
import shutil
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QFont, QColor, QPalette

from crystal_classifier import CrystalClassifier


class CrystalStageApp(QMainWindow):
    """Main application window for crystal stage recognition."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crystal Stage Recognition System")
        self.setMinimumSize(900, 700)

        # Initialize classifier
        self.classifier = None
        self.current_image_path = None
        self.continuous_mode = False
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer_tick)

        # Dataset path for simulation
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = os.path.join(self.base_dir, "dataset", "train")

        # Setup UI
        self._setup_ui()
        self._setup_styles()

        # Load model (in a real app, this might be async)
        self._load_model()

    def _setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title_label = QLabel("Crystal Stage Recognition System")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Main content area
        content_splitter = QSplitter(Qt.Horizontal)

        # Left panel - Image display
        left_panel = QGroupBox("Image Preview")
        left_layout = QVBoxLayout(left_panel)

        self.image_label = QLabel()
        self.image_label.setMinimumSize(400, 400)
        self.image_label.setMaximumSize(500, 500)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "QLabel { background-color: #f0f0f0; border: 2px dashed #ccc; border-radius: 8px; }"
        )
        self.image_label.setText("No image loaded\n\nClick 'Select Image' or 'Simulate Capture'")
        left_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        self.image_info_label = QLabel("")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.image_info_label)

        content_splitter.addWidget(left_panel)

        # Right panel - Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Prediction result
        result_group = QGroupBox("Prediction Result")
        result_layout = QVBoxLayout(result_group)

        self.stage_label = QLabel("Stage: --")
        self.stage_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.stage_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.stage_label)

        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setFont(QFont("Arial", 14))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.confidence_label)

        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setFormat("%v%")
        result_layout.addWidget(self.confidence_bar)

        right_layout.addWidget(result_group)

        # All probabilities
        probs_group = QGroupBox("Stage Probabilities")
        probs_layout = QVBoxLayout(probs_group)

        self.prob_bars = []
        self.prob_labels = []
        for i in range(6):
            h_layout = QHBoxLayout()
            label = QLabel(f"Stage {i}:")
            label.setFixedWidth(60)
            h_layout.addWidget(label)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(True)
            bar.setFormat("%v%")
            h_layout.addWidget(bar)

            self.prob_labels.append(label)
            self.prob_bars.append(bar)
            probs_layout.addLayout(h_layout)

        right_layout.addWidget(probs_group)

        # Manual correction section
        correct_group = QGroupBox("Manual Correction")
        correct_layout = QVBoxLayout(correct_group)

        correct_label = QLabel("If prediction is wrong, select correct stage:")
        correct_layout.addWidget(correct_label)

        correct_h_layout = QHBoxLayout()
        self.correct_combo = QComboBox()
        self.correct_combo.addItems([f"Stage {i}" for i in range(6)])
        self.correct_combo.setMinimumHeight(30)
        correct_h_layout.addWidget(self.correct_combo)

        self.save_correct_btn = QPushButton("Save Correction")
        self.save_correct_btn.setMinimumHeight(30)
        self.save_correct_btn.clicked.connect(self._on_save_correction)
        self.save_correct_btn.setEnabled(False)
        correct_h_layout.addWidget(self.save_correct_btn)

        correct_layout.addLayout(correct_h_layout)

        self.correct_status_label = QLabel("")
        self.correct_status_label.setStyleSheet("color: #666;")
        correct_layout.addWidget(self.correct_status_label)

        right_layout.addWidget(correct_group)
        right_layout.addStretch()

        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([500, 400])

        main_layout.addWidget(content_splitter)

        # Control buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(15)

        self.select_btn = QPushButton("Select Image")
        self.select_btn.setMinimumHeight(40)
        self.select_btn.clicked.connect(self._on_select_image)
        button_layout.addWidget(self.select_btn)

        self.simulate_btn = QPushButton("Simulate Capture")
        self.simulate_btn.setMinimumHeight(40)
        self.simulate_btn.clicked.connect(self._on_simulate_capture)
        button_layout.addWidget(self.simulate_btn)

        self.continuous_btn = QPushButton("Start Continuous")
        self.continuous_btn.setMinimumHeight(40)
        self.continuous_btn.setCheckable(True)
        self.continuous_btn.clicked.connect(self._on_toggle_continuous)
        button_layout.addWidget(self.continuous_btn)

        main_layout.addLayout(button_layout)

        # History
        history_group = QGroupBox("History")
        history_layout = QVBoxLayout(history_group)

        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(120)
        self.history_list.itemDoubleClicked.connect(self._on_history_item_clicked)
        history_layout.addWidget(self.history_list)

        main_layout.addWidget(history_group)

        # Status bar
        self.statusBar().showMessage("Ready")

    def _setup_styles(self):
        """Setup application styles."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #fafafa;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #4a90d9;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #357abd;
            }
            QPushButton:pressed {
                background-color: #2a5f8f;
            }
            QPushButton:checked {
                background-color: #e74c3c;
            }
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 4px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #4a90d9;
                border-radius: 3px;
            }
            QListWidget {
                border: 1px solid #ddd;
                border-radius: 4px;
            }
        """)

    def _load_model(self):
        """Load the classification model."""
        try:
            self.statusBar().showMessage("Loading model...")
            QApplication.processEvents()

            self.classifier = CrystalClassifier()

            self.statusBar().showMessage("Model loaded successfully")
        except FileNotFoundError as e:
            QMessageBox.critical(
                self, "Error",
                f"Model file not found:\n{e}\n\nPlease ensure the model has been trained."
            )
            self.statusBar().showMessage("Model not loaded")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")
            self.statusBar().showMessage("Model not loaded")

    def _on_select_image(self):
        """Handle select image button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", self.base_dir,
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)"
        )
        if file_path:
            self._load_and_predict(file_path)

    def _on_simulate_capture(self):
        """Handle simulate capture button click (random image from dataset)."""
        if not os.path.exists(self.dataset_dir):
            QMessageBox.warning(
                self, "Warning",
                f"Dataset directory not found:\n{self.dataset_dir}"
            )
            return

        # Collect all images from dataset
        all_images = []
        for stage_dir in range(6):
            stage_path = os.path.join(self.dataset_dir, str(stage_dir))
            if os.path.exists(stage_path):
                for img_file in os.listdir(stage_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        all_images.append(os.path.join(stage_path, img_file))

        if not all_images:
            QMessageBox.warning(self, "Warning", "No images found in dataset")
            return

        # Random select
        image_path = random.choice(all_images)
        self._load_and_predict(image_path)

    def _on_toggle_continuous(self, checked):
        """Handle continuous detection toggle."""
        if checked:
            self.continuous_btn.setText("Stop Continuous")
            self.continuous_mode = True
            self.timer.start(2000)  # Every 2 seconds
            self.statusBar().showMessage("Continuous detection started")
        else:
            self.continuous_btn.setText("Start Continuous")
            self.continuous_mode = False
            self.timer.stop()
            self.statusBar().showMessage("Continuous detection stopped")

    def _on_timer_tick(self):
        """Handle timer tick for continuous detection."""
        self._on_simulate_capture()

    def _on_history_item_clicked(self, item):
        """Handle history item double-click."""
        # Extract image path from item data
        image_path = item.data(Qt.UserRole)
        if image_path and os.path.exists(image_path):
            self._load_and_predict(image_path)

    def _load_and_predict(self, image_path: str):
        """Load an image and run prediction."""
        if self.classifier is None:
            QMessageBox.warning(self, "Warning", "Model not loaded")
            return

        self.current_image_path = image_path

        # Display image
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Warning", f"Failed to load image:\n{image_path}")
            return

        scaled_pixmap = pixmap.scaled(
            400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

        # Show image info
        file_name = os.path.basename(image_path)
        self.image_info_label.setText(f"{file_name} ({pixmap.width()}x{pixmap.height()})")

        # Run prediction
        self.statusBar().showMessage("Predicting...")
        QApplication.processEvents()

        try:
            stage, confidence, probs = self.classifier.predict(image_path)

            # Update result display
            self._update_results(stage, confidence, probs)

            # Add to history
            self._add_to_history(image_path, stage, confidence)

            self.statusBar().showMessage(f"Prediction complete: Stage {stage} ({confidence:.1f}%)")

            # Enable correction button and set combo to predicted stage
            self.save_correct_btn.setEnabled(True)
            self.correct_combo.setCurrentIndex(stage)
            self.correct_status_label.setText("")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prediction failed:\n{e}")
            self.statusBar().showMessage("Prediction failed")

    def _update_results(self, stage: int, confidence: float, probs: list):
        """Update the results display."""
        # Main result
        self.stage_label.setText(f"Stage {stage}")

        # Color based on confidence
        if confidence >= 80:
            color = "#27ae60"  # Green
        elif confidence >= 60:
            color = "#f39c12"  # Orange
        else:
            color = "#e74c3c"  # Red

        self.stage_label.setStyleSheet(f"color: {color};")
        self.confidence_label.setText(f"Confidence: {confidence:.1f}%")
        self.confidence_bar.setValue(int(confidence))

        # Update probability bars
        for i, (bar, prob) in enumerate(zip(self.prob_bars, probs)):
            bar.setValue(int(prob))

            # Highlight the predicted stage
            if i == stage:
                bar.setStyleSheet("""
                    QProgressBar::chunk { background-color: #27ae60; }
                """)
                self.prob_labels[i].setStyleSheet("font-weight: bold; color: #27ae60;")
            else:
                bar.setStyleSheet("")
                self.prob_labels[i].setStyleSheet("")

    def _add_to_history(self, image_path: str, stage: int, confidence: float):
        """Add a prediction to history."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        file_name = os.path.basename(image_path)

        item_text = f"[{timestamp}] {file_name} -> Stage {stage} ({confidence:.1f}%)"
        item = QListWidgetItem(item_text)
        item.setData(Qt.UserRole, image_path)

        # Color based on confidence
        if confidence >= 80:
            item.setForeground(QColor("#27ae60"))
        elif confidence >= 60:
            item.setForeground(QColor("#f39c12"))
        else:
            item.setForeground(QColor("#e74c3c"))

        self.history_list.insertItem(0, item)

        # Limit history size
        while self.history_list.count() > 50:
            self.history_list.takeItem(self.history_list.count() - 1)

    def _on_save_correction(self):
        """Save the current image to the selected stage folder."""
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "No image loaded")
            return

        if not os.path.exists(self.current_image_path):
            QMessageBox.warning(self, "Warning", "Image file no longer exists")
            return

        # Get selected stage
        correct_stage = self.correct_combo.currentIndex()
        target_dir = os.path.join(self.dataset_dir, str(correct_stage))

        # Create directory if not exists
        os.makedirs(target_dir, exist_ok=True)

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = os.path.basename(self.current_image_path)
        name, ext = os.path.splitext(original_name)
        new_filename = f"{name}_{timestamp}{ext}"
        target_path = os.path.join(target_dir, new_filename)

        try:
            # Copy image to target folder
            shutil.copy2(self.current_image_path, target_path)

            # Update status
            self.correct_status_label.setText(f"Saved to Stage {correct_stage} folder")
            self.correct_status_label.setStyleSheet("color: #27ae60;")
            self.statusBar().showMessage(f"Image saved to {target_path}")

            # Count images in target folder
            count = len([f for f in os.listdir(target_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            self.correct_status_label.setText(
                f"Saved to Stage {correct_stage} folder (now {count} images)")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save image:\n{e}")
            self.correct_status_label.setText("Save failed")
            self.correct_status_label.setStyleSheet("color: #e74c3c;")

    def closeEvent(self, event):
        """Handle window close event."""
        if self.continuous_mode:
            self.timer.stop()
        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("Crystal Stage Recognition")

    window = CrystalStageApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

"""
Crystal Stage Recognition System - GUI Application
A PyQt5-based desktop application for crystal stage classification.
Enhanced UI for demonstration purposes.
"""

import os
import sys
import random
import shutil
from datetime import datetime
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QFileDialog, QListWidget, QListWidgetItem,
    QProgressBar, QGroupBox, QFrame, QMessageBox, QSplitter, QComboBox,
    QGraphicsDropShadowEffect
)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QPixmap, QFont, QColor, QPalette, QLinearGradient, QPainter

from crystal_classifier import CrystalClassifier


# Color scheme
COLORS = {
    'primary': '#2563eb',       # Blue
    'primary_dark': '#1d4ed8',
    'primary_light': '#3b82f6',
    'success': '#10b981',       # Green
    'warning': '#f59e0b',       # Orange
    'danger': '#ef4444',        # Red
    'background': '#f8fafc',
    'card': '#ffffff',
    'text': '#1e293b',
    'text_secondary': '#64748b',
    'border': '#e2e8f0',
}


class HeaderWidget(QFrame):
    """Custom header widget with gradient background."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(70)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 30, 0)

        # Logo
        logo_label = QLabel()
        logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.jpg")
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path)
            scaled_logo = logo_pixmap.scaledToHeight(60, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_logo)
            logo_label.setStyleSheet("background-color: white; padding: 3px; border-radius: 3px;")
        layout.addWidget(logo_label)
        layout.addSpacing(15)

        # Title
        title = QLabel("反应自动控制系统")
        title.setFont(QFont("Microsoft YaHei UI", 16, QFont.Bold))
        title.setStyleSheet("color: white;")
        layout.addWidget(title)
        layout.addStretch()

        # Status indicator
        self.status_label = QLabel("就绪")
        self.status_label.setFont(QFont("Microsoft YaHei UI", 10))
        self.status_label.setStyleSheet("""
            color: white;
            background-color: rgba(255, 255, 255, 0.2);
            padding: 6px 12px;
            border-radius: 12px;
        """)
        layout.addWidget(self.status_label)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        gradient = QLinearGradient(0, 0, self.width(), 0)
        gradient.setColorAt(0, QColor(COLORS['primary']))
        gradient.setColorAt(1, QColor(COLORS['primary_dark']))

        painter.fillRect(self.rect(), gradient)

    def set_status(self, text, status_type='normal'):
        self.status_label.setText(text)
        if status_type == 'success':
            bg = 'rgba(16, 185, 129, 0.3)'
        elif status_type == 'warning':
            bg = 'rgba(245, 158, 11, 0.3)'
        elif status_type == 'error':
            bg = 'rgba(239, 68, 68, 0.3)'
        else:
            bg = 'rgba(255, 255, 255, 0.2)'

        self.status_label.setStyleSheet(f"""
            color: white;
            background-color: {bg};
            padding: 6px 12px;
            border-radius: 12px;
        """)


class StageResultWidget(QFrame):
    """Large stage result display widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(45)
        self.stage = None
        self.confidence = 0

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(0)

        self.stage_label = QLabel("--")
        self.stage_label.setFont(QFont("Microsoft YaHei", 16))
        self.stage_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.stage_label)

        self.confidence_label = QLabel("等待图像...")
        self.confidence_label.setFont(QFont("Microsoft YaHei", 9))
        self.confidence_label.setAlignment(Qt.AlignCenter)
        self.confidence_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.confidence_label)

        self._update_style()

    def set_result(self, stage: int, confidence: float):
        self.stage = stage
        self.confidence = confidence
        self.stage_label.setText(f"阶段{stage}")
        self.confidence_label.setText(f"置信度: {confidence:.1f}%")
        self._update_style()

    def _update_style(self):
        if self.stage is None:
            color = COLORS['text_secondary']
            bg = COLORS['card']
        elif self.confidence >= 80:
            color = COLORS['success']
            bg = '#ecfdf5'
        elif self.confidence >= 60:
            color = COLORS['warning']
            bg = '#fffbeb'
        else:
            color = COLORS['danger']
            bg = '#fef2f2'

        self.stage_label.setStyleSheet(f"color: {color};")
        self.setStyleSheet(f"""
            StageResultWidget {{
                background-color: {bg};
                border-radius: 12px;
                border: 2px solid {color if self.stage is not None else COLORS['border']};
            }}
        """)


class CrystalStageApp(QMainWindow):
    """Main application window for crystal stage recognition."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("反应自动控制系统")
        self.setMinimumSize(1000, 700)

        # Initialize classifier
        self.classifier = None
        self.current_image_path = None
        self.continuous_mode = False
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer_tick)

        # Statistics
        self.total_predictions = 0
        self.stage_counts = [0] * 6

        # Dataset path for simulation
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_dir = os.path.join(self.base_dir, "dataset", "train")

        # Setup UI
        self._setup_ui()
        self._setup_styles()

        # Load model
        self._load_model()

    def _setup_ui(self):
        """Setup the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Header
        self.header = HeaderWidget()
        main_layout.addWidget(self.header)

        # Content area
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setSpacing(20)
        content_layout.setContentsMargins(20, 20, 20, 20)

        # Left panel - Image display
        left_panel = self._create_left_panel()
        content_layout.addWidget(left_panel, stretch=3)

        # Right panel - Results
        right_panel = self._create_right_panel()
        content_layout.addWidget(right_panel, stretch=2)

        main_layout.addWidget(content_widget)

    def _create_left_panel(self):
        """Create the left panel with image preview and controls."""
        panel = QFrame()
        panel.setObjectName("leftPanel")
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        # Image card
        image_card = QFrame()
        image_card.setObjectName("imageCard")
        image_layout = QVBoxLayout(image_card)
        image_layout.setContentsMargins(20, 20, 20, 20)

        # Image label
        self.image_label = QLabel()
        self.image_label.setMinimumSize(450, 400)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setObjectName("imageLabel")
        self.image_label.setText("未加载图像\n\n选择图像或使用模拟拍照")
        image_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # Image info
        self.image_info_label = QLabel("")
        self.image_info_label.setAlignment(Qt.AlignCenter)
        self.image_info_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        image_layout.addWidget(self.image_info_label)

        layout.addWidget(image_card, stretch=1)  # 图像区域占用更多空间

        # Control buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.select_btn = QPushButton("选择图像")
        self.select_btn.setObjectName("primaryBtn")
        self.select_btn.setMinimumHeight(45)
        self.select_btn.clicked.connect(self._on_select_image)
        button_layout.addWidget(self.select_btn)

        self.simulate_btn = QPushButton("模拟拍照")
        self.simulate_btn.setObjectName("secondaryBtn")
        self.simulate_btn.setMinimumHeight(45)
        self.simulate_btn.clicked.connect(self._on_simulate_capture)
        button_layout.addWidget(self.simulate_btn)

        self.continuous_btn = QPushButton("开始自动模式")
        self.continuous_btn.setObjectName("accentBtn")
        self.continuous_btn.setMinimumHeight(45)
        self.continuous_btn.setCheckable(True)
        self.continuous_btn.clicked.connect(self._on_toggle_continuous)
        button_layout.addWidget(self.continuous_btn)

        layout.addLayout(button_layout)

        # History
        history_card = QFrame()
        history_card.setObjectName("historyCard")
        history_layout = QVBoxLayout(history_card)

        history_title = QLabel("检测历史")
        history_title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        history_layout.addWidget(history_title)

        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(150)
        self.history_list.itemDoubleClicked.connect(self._on_history_item_clicked)
        history_layout.addWidget(self.history_list)

        layout.addWidget(history_card)

        return panel

    def _create_right_panel(self):
        """Create the right panel with results and correction."""
        panel = QFrame()
        panel.setObjectName("rightPanel")
        layout = QVBoxLayout(panel)
        layout.setSpacing(15)

        # Result card
        result_card = QFrame()
        result_card.setObjectName("resultCard")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(20, 20, 20, 20)

        result_title = QLabel("识别结果")
        result_title.setFont(QFont("Microsoft YaHei", 14, QFont.Bold))
        result_layout.addWidget(result_title)

        self.result_widget = StageResultWidget()
        result_layout.addWidget(self.result_widget)

        layout.addWidget(result_card)

        # Probabilities card
        probs_card = QFrame()
        probs_card.setObjectName("probsCard")
        probs_card.setMinimumHeight(200)  # 确保足够高度
        probs_layout = QVBoxLayout(probs_card)
        probs_layout.setContentsMargins(15, 8, 15, 8)
        probs_layout.setSpacing(2)  # 减小行间距

        probs_title = QLabel("阶段概率")
        probs_title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        probs_layout.addWidget(probs_title)

        self.prob_bars = []
        self.prob_labels = []
        self.prob_value_labels = []

        for i in range(6):
            h_layout = QHBoxLayout()
            h_layout.setSpacing(8)
            h_layout.setContentsMargins(0, 0, 0, 0)  # 去掉额外间距

            label = QLabel(f"阶段{i}")
            label.setFixedWidth(80)
            label.setFont(QFont("Microsoft YaHei", 12))
            h_layout.addWidget(label, 0)  # stretch=0, 不拉伸

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(20)
            bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            h_layout.addWidget(bar, 1)  # stretch=1, 占用剩余空间

            value_label = QLabel("0%")
            value_label.setMinimumWidth(50)
            value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value_label.setFont(QFont("Microsoft YaHei", 12))
            value_label.setFixedWidth(60) 
            h_layout.addWidget(value_label, 0)  # stretch=0, 不拉伸

            self.prob_labels.append(label)
            self.prob_bars.append(bar)
            self.prob_value_labels.append(value_label)
            probs_layout.addLayout(h_layout)

        layout.addWidget(probs_card)

        # Manual correction card
        correct_card = QFrame()
        correct_card.setObjectName("correctCard")
        correct_layout = QVBoxLayout(correct_card)
        correct_layout.setContentsMargins(20, 15, 20, 15)

        correct_title = QLabel("手动修正")
        correct_title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        correct_layout.addWidget(correct_title)

        correct_desc = QLabel("如果预测错误，请选择正确的阶段：")
        correct_desc.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        correct_desc.setWordWrap(True)
        correct_layout.addWidget(correct_desc)

        correct_h_layout = QHBoxLayout()
        correct_h_layout.setSpacing(10)

        self.correct_combo = QComboBox()
        self.correct_combo.addItems([f"阶段{i}" for i in range(6)])
        self.correct_combo.setMinimumHeight(36)
        correct_h_layout.addWidget(self.correct_combo)

        self.save_correct_btn = QPushButton("保存")
        self.save_correct_btn.setObjectName("successBtn")
        self.save_correct_btn.setMinimumHeight(36)
        self.save_correct_btn.setFixedWidth(80)
        self.save_correct_btn.clicked.connect(self._on_save_correction)
        self.save_correct_btn.setEnabled(False)
        correct_h_layout.addWidget(self.save_correct_btn)

        correct_layout.addLayout(correct_h_layout)

        self.correct_status_label = QLabel("")
        self.correct_status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        correct_layout.addWidget(self.correct_status_label)

        layout.addWidget(correct_card)

        # Statistics card
        stats_card = QFrame()
        stats_card.setObjectName("statsCard")
        stats_layout = QVBoxLayout(stats_card)
        stats_layout.setContentsMargins(20, 15, 20, 15)

        stats_title = QLabel("会话统计")
        stats_title.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        stats_layout.addWidget(stats_title)

        self.stats_label = QLabel("总预测次数: 0")
        self.stats_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        stats_layout.addWidget(self.stats_label)

        layout.addWidget(stats_card)

        return panel

    def _setup_styles(self):
        """Setup application styles."""
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {COLORS['background']};
            }}

            #leftPanel, #rightPanel {{
                background-color: transparent;
            }}

            #imageCard, #resultCard, #probsCard, #correctCard, #historyCard, #statsCard {{
                background-color: {COLORS['card']};
                border-radius: 12px;
                border: 1px solid {COLORS['border']};
            }}

            #imageLabel {{
                background-color: {COLORS['background']};
                border: 2px dashed {COLORS['border']};
                border-radius: 8px;
                color: {COLORS['text_secondary']};
                font-size: 13px;
            }}

            QPushButton {{
                font-family: 'Microsoft YaHei';
                font-size: 13px;
                font-weight: 600;
                border-radius: 8px;
                padding: 8px 20px;
            }}

            #primaryBtn {{
                background-color: {COLORS['primary']};
                color: white;
                border: none;
            }}
            #primaryBtn:hover {{
                background-color: {COLORS['primary_dark']};
            }}

            #secondaryBtn {{
                background-color: white;
                color: {COLORS['primary']};
                border: 2px solid {COLORS['primary']};
            }}
            #secondaryBtn:hover {{
                background-color: {COLORS['background']};
            }}

            #accentBtn {{
                background-color: {COLORS['success']};
                color: white;
                border: none;
            }}
            #accentBtn:hover {{
                background-color: #059669;
            }}
            #accentBtn:checked {{
                background-color: {COLORS['danger']};
            }}

            #successBtn {{
                background-color: {COLORS['success']};
                color: white;
                border: none;
            }}
            #successBtn:hover {{
                background-color: #059669;
            }}
            #successBtn:disabled {{
                background-color: {COLORS['border']};
                color: {COLORS['text_secondary']};
            }}

            QProgressBar {{
                border: none;
                border-radius: 8px;
                background-color: {COLORS['background']};
            }}
            QProgressBar::chunk {{
                border-radius: 8px;
                background-color: {COLORS['primary']};
            }}
            QProgressBar[state="normal"]::chunk {{
                background-color: {COLORS['primary']};
            }}
            QProgressBar[state="pred"]::chunk {{
                background-color: {COLORS['success']};
            }}
            QComboBox {{
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                padding: 6px 12px;
                background-color: white;
                font-size: 12px;
            }}
            QComboBox:hover {{
                border-color: {COLORS['primary']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 30px;
            }}

            QListWidget {{
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                background-color: white;
                font-size: 11px;
            }}
            QListWidget::item {{
                padding: 6px;
                border-bottom: 1px solid {COLORS['border']};
            }}
            QListWidget::item:hover {{
                background-color: {COLORS['background']};
            }}
        """)

    def _load_model(self):
        """Load the classification model."""
        try:
            self.header.set_status("正在加载模型...", 'warning')
            QApplication.processEvents()

            self.classifier = CrystalClassifier()

            self.header.set_status("模型就绪", 'success')
        except FileNotFoundError as e:
            QMessageBox.critical(
                self, "错误",
                f"模型文件未找到:\n{e}\n\n请确保模型已训练完成。"
            )
            self.header.set_status("模型错误", 'error')
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型加载失败:\n{e}")
            self.header.set_status("模型错误", 'error')

    def _on_select_image(self):
        """Handle select image button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图像", self.base_dir,
            "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*)"
        )
        if file_path:
            self._load_and_predict(file_path)

    def _on_simulate_capture(self):
        """Handle simulate capture button click (random image from dataset)."""
        if not os.path.exists(self.dataset_dir):
            QMessageBox.warning(
                self, "警告",
                f"数据集目录未找到:\n{self.dataset_dir}"
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
            QMessageBox.warning(self, "警告", "数据集中未找到图像")
            return

        # Random select
        image_path = random.choice(all_images)
        self._load_and_predict(image_path)

    def _on_toggle_continuous(self, checked):
        """Handle continuous detection toggle."""
        if checked:
            self.continuous_btn.setText("停止自动模式")
            self.continuous_mode = True
            self.timer.start(2500)
            self.header.set_status("自动模式运行中", 'warning')
        else:
            self.continuous_btn.setText("开始自动模式")
            self.continuous_mode = False
            self.timer.stop()
            self.header.set_status("就绪", 'success')

    def _on_timer_tick(self):
        """Handle timer tick for continuous detection."""
        self._on_simulate_capture()

    def _on_history_item_clicked(self, item):
        """Handle history item double-click."""
        image_path = item.data(Qt.UserRole)
        if image_path and os.path.exists(image_path):
            self._load_and_predict(image_path)

    def _load_and_predict(self, image_path: str):
        """Load an image and run prediction."""
        if self.classifier is None:
            QMessageBox.warning(self, "警告", "模型未加载")
            return

        self.current_image_path = image_path

        # Display image
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "警告", f"图像加载失败:\n{image_path}")
            return

        scaled_pixmap = pixmap.scaled(
            450, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

        # Show image info
        file_name = os.path.basename(image_path)
        self.image_info_label.setText(f"{file_name} ({pixmap.width()} x {pixmap.height()})")

        # Run prediction
        self.header.set_status("分析中...", 'warning')
        QApplication.processEvents()

        try:
            stage, confidence, probs = self.classifier.predict(image_path)

            # Update result display
            self._update_results(stage, confidence, probs)

            # Add to history
            self._add_to_history(image_path, stage, confidence)

            # Update statistics
            self.total_predictions += 1
            self.stage_counts[stage] += 1
            self._update_stats()

            # Enable correction
            self.save_correct_btn.setEnabled(True)
            self.correct_combo.setCurrentIndex(stage)
            self.correct_status_label.setText("")

            # Update header status
            if confidence >= 80:
                self.header.set_status(f"阶段{stage} ({confidence:.0f}%)", 'success')
            elif confidence >= 60:
                self.header.set_status(f"阶段{stage} ({confidence:.0f}%)", 'warning')
            else:
                self.header.set_status(f"阶段{stage} ({confidence:.0f}%) - 置信度低", 'error')

        except Exception as e:
            QMessageBox.critical(self, "错误", f"预测失败:\n{e}")
            self.header.set_status("预测错误", 'error')

    def _update_results(self, stage: int, confidence: float, probs: list):
        """Update the results display."""
        self.result_widget.set_result(stage, confidence)

        # Update probability bars
        for i, (bar, label, value_label, prob) in enumerate(
            zip(self.prob_bars, self.prob_labels, self.prob_value_labels, probs)
        ):
            bar.setValue(int(prob))
            value_label.setText(f"{prob:.0f}%")

            # Highlight the predicted stage
            if i == stage:
                bar.setProperty("state", "pred")
                label.setStyleSheet(f"font-weight: bold; color: {COLORS['success']};")
                value_label.setStyleSheet(f"font-weight: bold; color: {COLORS['success']};")
            else:
                bar.setProperty("state", "normal")
                label.setStyleSheet("")
                value_label.setStyleSheet("")
            
            # 强制刷新QSS（必须）
            bar.style().unpolish(bar)
            bar.style().polish(bar)
            bar.update()

    def _add_to_history(self, image_path: str, stage: int, confidence: float):
        """Add a prediction to history."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        file_name = os.path.basename(image_path)

        item_text = f"[{timestamp}] {file_name} → 阶段{stage} ({confidence:.0f}%)"
        item = QListWidgetItem(item_text)
        item.setData(Qt.UserRole, image_path)

        if confidence >= 80:
            item.setForeground(QColor(COLORS['success']))
        elif confidence >= 60:
            item.setForeground(QColor(COLORS['warning']))
        else:
            item.setForeground(QColor(COLORS['danger']))

        self.history_list.insertItem(0, item)

        while self.history_list.count() > 50:
            self.history_list.takeItem(self.history_list.count() - 1)

    def _update_stats(self):
        """Update statistics display."""
        stats_text = f"总预测次数: {self.total_predictions}\n"
        if self.total_predictions > 0:
            stage_stats = [f"阶段{i}: {c}" for i, c in enumerate(self.stage_counts) if c > 0]
            stats_text += "分布: " + " | ".join(stage_stats)
        self.stats_label.setText(stats_text)

    def _on_save_correction(self):
        """Save the current image to the selected stage folder."""
        if not self.current_image_path:
            QMessageBox.warning(self, "警告", "未加载图像")
            return

        if not os.path.exists(self.current_image_path):
            QMessageBox.warning(self, "警告", "图像文件已不存在")
            return

        correct_stage = self.correct_combo.currentIndex()
        target_dir = os.path.join(self.dataset_dir, str(correct_stage))
        os.makedirs(target_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_name = os.path.basename(self.current_image_path)
        name, ext = os.path.splitext(original_name)
        new_filename = f"{name}_{timestamp}{ext}"
        target_path = os.path.join(target_dir, new_filename)

        try:
            shutil.copy2(self.current_image_path, target_path)

            count = len([f for f in os.listdir(target_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])

            self.correct_status_label.setText(
                f"已保存到阶段{correct_stage} (共{count}张图像)")
            self.correct_status_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 11px;")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"图像保存失败:\n{e}")
            self.correct_status_label.setText("保存失败")
            self.correct_status_label.setStyleSheet(f"color: {COLORS['danger']}; font-size: 11px;")

    def closeEvent(self, event):
        """Handle window close event."""
        if self.continuous_mode:
            self.timer.stop()
        event.accept()


def main():
    """Main entry point."""
    # 启用高DPI支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("反应自动控制系统")
    app.setStyle('Fusion')

    window = CrystalStageApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

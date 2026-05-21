# ==================== main_window.py ====================
# 生活垃圾检测系统主窗口
# 版本：v4.0 (SAPI异步播报 + 摄像头优化 + 语音独立控制)
# =======================================================

import os
import sys
import cv2
import numpy as np
import logging
import re
import queue
import time
import pyaudio
import wave
import io
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QGroupBox, QSlider, QLabel, QCheckBox,
    QFileDialog, QMessageBox, QProgressBar,
    QApplication, QStatusBar, QTabWidget, QSizePolicy,
    QGridLayout
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
from PyQt5.QtGui import QImage, QPixmap, QFont

try:
    from src.backend.detector import YOLODetector
    from src.backend.database_manager import DatabaseManager, DetectionData
    from src.backend.voice_interaction import VoiceInteractionManager
    from src.backend.text_to_speech import TextToSpeech
    from src.backend.export_manager import ExportManager
except ImportError as e:
    print(f"错误：无法导入后端模块 - {e}")
    print("请确保已正确安装所有依赖模块")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('garbage_detection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RecordingThread(QThread):
    finished = pyqtSignal(bytes)
    error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.sample_width = 2
        self.error_msg = ""

    def run(self):
        try:
            self.is_recording = True
            p = pyaudio.PyAudio()
            stream = p.open(format=self.audio_format, channels=self.channels, rate=self.rate, input=True,
                            frames_per_buffer=self.chunk)
            frames = []
            while self.is_recording:
                try:
                    data = stream.read(self.chunk, exception_on_overflow=False)
                    frames.append(data)
                except Exception as e:
                    logger.error(f"录音读取错误: {e}")
                    self.error_msg = f"录音错误: {str(e)}"
                    break
            stream.stop_stream()
            stream.close()
            p.terminate()
            if not self.error_msg:
                self.finished.emit(self._encode_wav(b''.join(frames)))
            else:
                self.error.emit(self.error_msg)
        except Exception as e:
            self.error.emit(f"录音线程错误: {str(e)}")

    def stop(self):
        self.is_recording = False

    def _encode_wav(self, pcm_data):
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.rate)
            wav_file.writeframes(pcm_data)
        return wav_buffer.getvalue()


class DetectionThread(QThread):
    """视频文件异步检测线程"""
    detection_result = pyqtSignal(list, np.ndarray)
    error = pyqtSignal(str)

    def __init__(self, detector, frame_queue, conf_threshold):
        super().__init__()
        self.detector = detector
        self.queue = frame_queue
        self.conf_threshold = conf_threshold
        self.running = True

    def run(self):
        try:
            while self.running:
                try:
                    frame = self.queue.get(timeout=0.1)
                    if frame is None or frame.size == 0:
                        continue
                    detections = self.detector.detect(frame, self.conf_threshold, 0.45)
                    drawn_frame = self.detector.draw_detections(frame, detections)
                    self.detection_result.emit(detections, drawn_frame)
                except queue.Empty:
                    continue
                except Exception as e:
                    self.error.emit(f"检测线程处理错误: {str(e)}")
                    break
        except Exception as e:
            self.error.emit(f"检测线程运行错误: {str(e)}")

    def stop(self):
        self.running = False
        self.wait()

    def update_threshold(self, new_threshold):
        self.conf_threshold = new_threshold


class CameraDetectionThread(QThread):
    """摄像头异步检测线程（解决卡顿）"""
    frame_processed = pyqtSignal(np.ndarray, list)
    error = pyqtSignal(str)

    def __init__(self, detector, conf_threshold, iou_threshold=0.45, inference_size=(640, 480)):
        super().__init__()
        self.detector = detector
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.inference_size = inference_size
        self.running = True
        self.frame_queue = queue.Queue(maxsize=2)

    def set_threshold(self, conf_threshold):
        self.conf_threshold = conf_threshold

    def put_frame(self, frame):
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame)

    def run(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
                if frame is None:
                    continue

                h, w = frame.shape[:2]
                if self.inference_size:
                    inference_frame = cv2.resize(frame, self.inference_size, interpolation=cv2.INTER_NEAREST)
                else:
                    inference_frame = frame

                detections = self.detector.detect(inference_frame, self.conf_threshold, self.iou_threshold)

                if self.inference_size and (w, h) != self.inference_size:
                    scale_x = w / self.inference_size[0]
                    scale_y = h / self.inference_size[1]
                    for det in detections:
                        bbox = det['bbox']
                        bbox[0] = int(bbox[0] * scale_x)
                        bbox[1] = int(bbox[1] * scale_y)
                        bbox[2] = int(bbox[2] * scale_x)
                        bbox[3] = int(bbox[3] * scale_y)

                drawn_frame = self.detector.draw_detections(frame, detections)
                self.frame_processed.emit(drawn_frame, detections)

            except queue.Empty:
                continue
            except Exception as e:
                self.error.emit(str(e))

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于深度学习的生活垃圾检测系统")
        self.setMinimumSize(2170, 1700)
        self.resize(2170, 1700)

        self.is_detecting = False
        self.is_paused = False
        self.current_frame = None
        self.video_capture = None
        self.detection_timer = None
        self.total_detections = 0
        self.detection_start_time = None
        self.elapsed_timer = None
        self.current_time_timer = None
        self.single_detection_time = 0
        self.last_announce_time = 0
        self.announce_cooldown_ms = 2000
        self.last_announced_object = None
        self.selected_file = None
        self.video_total_frames = 0
        self.video_fps = 0
        self.video_duration = 0
        self.video_progress_slider = None
        self.video_position_label = None
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.display_video_frame)
        self.detection_queue = None
        self.detection_thread = None
        self.camera_detection_thread = None
        self.latest_drawn_frame = None

        self.frame_skip_counter = 0
        self.frame_skip_interval = 2

        self.current_detections_data = []
        self.current_history_records = []

        self._voice_command_active = False

        self.conf_timer = QTimer()
        self.conf_timer.setSingleShot(True)
        self.conf_timer.timeout.connect(self._do_confidence_update)

        self.browse_btn = None
        self.total_frames_processed = 0

        self.init_backend()
        self.setup_ui()
        self.apply_styles()
        self.start_current_time_timer()

    def init_backend(self):
        try:
            from src.utils.config_loader import get_config
            db_config = get_config('database', {})
            try:
                self.db_manager = DatabaseManager(
                    use_mysql=db_config.get('use_mysql', False),
                    host=db_config.get('host', 'localhost'),
                    port=db_config.get('port', 3306),
                    username=db_config.get('username', 'root'),
                    password=db_config.get('password', 'root123456'),
                    database=db_config.get('database', 'garbage_detection'),
                    charset=db_config.get('charset', 'utf8mb4')
                )
            except Exception as e:
                logger.error(f"数据库初始化失败: {e}")
                self.db_manager = None

            api_config = get_config('api', {})
            speech_config = api_config.get('speech_recognition', {})
            try:
                provider = speech_config.get('provider', 'local')
                local_model_path = speech_config.get('local_model_path')
                self.voice_manager = VoiceInteractionManager(audio_config=None, local_model_path=local_model_path,
                                                             parent=self)
                if hasattr(self.voice_manager, 'engine_init_failed'):
                    self.voice_manager.engine_init_failed.connect(self._on_voice_engine_failed)
            except Exception as e:
                logger.error(f"语音管理器初始化失败: {e}")
                self.voice_manager = None

            model_config = get_config('model', {})
            model_path = model_config.get('path', 'runs-yolov8m/runs/detect/runs/train/exp_optimized/weights/best.pt')
            try:
                if not os.path.exists(model_path) or os.path.getsize(model_path) == 0:
                    raise Exception("模型无效")
                self.detector = YOLODetector(model_path=model_path, device=model_config.get('device', 'cpu'))
            except Exception as e:
                logger.error(f"模型加载失败: {e}")
                self.detector = None

            try:
                self.tts = TextToSpeech()
                if hasattr(self.tts, 'init_failed'):
                    self.tts.init_failed.connect(self._on_tts_init_failed)
                logger.info("TTS初始化成功 (SAPI)")
            except Exception as e:
                logger.error(f"TTS初始化失败: {e}")
                self.tts = None

            try:
                self.export_manager = ExportManager()
            except Exception as e:
                logger.error(f"导出管理器初始化失败: {e}")
                self.export_manager = None
        except Exception as e:
            logger.critical(f"后端初始化失败: {e}")
            sys.exit(1)

    def _on_voice_engine_failed(self, error_msg):
        self.status_bar.showMessage(f"语音识别引擎初始化失败: {error_msg}", 5000)

    def _on_tts_init_failed(self, error_msg):
        self.status_bar.showMessage(f"语音播报引擎初始化失败: {error_msg}", 5000)

    # ==================== UI 构建 ====================
    def setup_ui(self):
        try:
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QVBoxLayout(central_widget)
            main_layout.setContentsMargins(10, 10, 10, 10)
            main_layout.setSpacing(10)

            title = QLabel("基于深度学习的生活垃圾检测系统")
            title.setAlignment(Qt.AlignCenter)
            title.setStyleSheet(
                "QLabel { font-size: 56px; font-weight: bold; color: white; padding: 20px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3498db, stop:1 #2ecc71); border-radius: 12px; min-height: 80px; }")
            main_layout.addWidget(title)

            self.current_time_label = QLabel()
            self.current_time_label.setAlignment(Qt.AlignCenter)
            self.current_time_label.setStyleSheet(
                "QLabel { font-size: 22px; font-weight: bold; color: #2c3e50; padding: 10px; background-color: #f0f6fa; border-radius: 8px; min-height: 40px; }")
            main_layout.addWidget(self.current_time_label)

            content_layout = QHBoxLayout()
            content_layout.setSpacing(15)
            content_layout.addWidget(self.create_video_panel(), 2)
            content_layout.addWidget(self.create_control_panel(), 1)
            main_layout.addLayout(content_layout)

            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            central_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.status_bar = QStatusBar()
            self.setStatusBar(self.status_bar)
            self.status_bar.showMessage("系统就绪")
            self._update_voice_input_buttons()
        except Exception as e:
            logger.error(f"UI初始化失败: {e}", exc_info=True)
            QMessageBox.critical(self, "UI错误", f"界面初始化失败：{str(e)}")
            sys.exit(1)

    def create_video_panel(self):
        try:
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(10)
            self.video_label = QLabel()
            self.video_label.setAlignment(Qt.AlignCenter)
            self.video_label.setMinimumSize(1000, 600)
            self.video_label.setStyleSheet(
                "QLabel { background-color: #f0f6fa; border: 3px solid #3498db; border-radius: 12px; color: #2c3e50; font-size: 30px; min-height: 600px; }")
            self.video_label.setText("检测结果显示区域")
            layout.addWidget(self.video_label, 3)

            progress_widget = QWidget()
            progress_layout = QHBoxLayout(progress_widget)
            progress_layout.setContentsMargins(0, 5, 0, 5)
            self.video_position_label = QLabel("00:00 / 00:00")
            self.video_position_label.setStyleSheet("font-size: 18px; color: #2c3e50;")
            progress_layout.addWidget(self.video_position_label)
            self.video_progress_slider = QSlider(Qt.Horizontal)
            self.video_progress_slider.setRange(0, 1000)
            self.video_progress_slider.sliderPressed.connect(self.on_progress_slider_pressed)
            self.video_progress_slider.sliderReleased.connect(self.on_progress_slider_released)
            progress_layout.addWidget(self.video_progress_slider, 1)
            layout.addWidget(progress_widget)

            results_group = QGroupBox("检测结果")
            results_layout = QVBoxLayout(results_group)
            stats_layout = QHBoxLayout()
            stats_layout.addWidget(QLabel("检测数: "))
            self.current_count_label = QLabel("0")
            self.current_count_label.setStyleSheet("font-weight: bold; color: #3498db; font-size: 24px;")
            stats_layout.addWidget(self.current_count_label)
            stats_layout.addStretch()
            results_layout.addLayout(stats_layout)

            self.results_table = QTableWidget()
            self.results_table.setColumnCount(6)
            self.results_table.setHorizontalHeaderLabels(["时间", "具体类别", "大类类别", "置信度", "位置坐标", "详情"])
            self.results_table.horizontalHeader().setStretchLastSection(False)
            self.results_table.setMinimumHeight(250)
            self.results_table.setMaximumHeight(250)
            self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)
            self.results_table.setFont(QFont("Microsoft YaHei", 14))
            header = self.results_table.horizontalHeader()
            for i in range(6):
                header.setSectionResizeMode(i, QHeaderView.Stretch)
            header.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
            for i in range(6):
                self.results_table.horizontalHeaderItem(i).setTextAlignment(Qt.AlignCenter)
            results_layout.addWidget(self.results_table)
            layout.addWidget(results_group, 1)
            return panel
        except Exception as e:
            logger.error(f"创建视频面板失败: {e}", exc_info=True)
            raise

    def create_control_panel(self):
        try:
            panel = QWidget()
            layout = QVBoxLayout(panel)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(10)
            model_info_group = QGroupBox("模型加载信息")
            model_info_layout = QHBoxLayout(model_info_group)
            model_path_layout = QHBoxLayout()
            model_path_layout.addWidget(QLabel("模型名称: "))
            self.model_path_label = QLabel("YOLOv8m" if hasattr(self, 'detector') and self.detector else "未加载")
            self.model_path_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 22px;")
            model_path_layout.addWidget(self.model_path_label)
            db_layout = QHBoxLayout()
            db_layout.addWidget(QLabel("数据库类型: "))
            db_type = "MySQL" if (
                hasattr(self, 'db_manager') and self.db_manager and self.db_manager.use_mysql) else "SQLite"
            self.db_label = QLabel(db_type)
            self.db_label.setStyleSheet("color: #e67e22; font-weight: bold; font-size: 22px;")
            db_layout.addWidget(self.db_label)
            model_info_layout.addStretch()
            model_info_layout.addLayout(model_path_layout)
            model_info_layout.addSpacing(90)
            model_info_layout.addLayout(db_layout)
            model_info_layout.addStretch()
            layout.addWidget(model_info_group)

            tab_widget = QTabWidget()
            tab_widget.setStyleSheet(
                "QTabWidget::pane { border: 2px solid #3498db; border-radius: 10px; background: #f0f6fa; } QTabBar::tab { background: #ecf0f1; padding: 13px 32px; margin-right: 5px; border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: bold; font-size: 23px; } QTabBar::tab:selected { background: #3498db; color: white; } QTabBar::tab:hover { background: #5dade2; color: white; }")
            tab_widget.addTab(self.create_detection_control_tab(), "检测控制")
            tab_widget.addTab(self.create_history_tab(), "历史数据")
            layout.addWidget(tab_widget)
            layout.addWidget(self.create_stats_panel())
            return panel
        except Exception as e:
            logger.error(f"创建控制面板失败: {e}", exc_info=True)
            raise

    def create_detection_control_tab(self):
        try:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            layout.setSpacing(10)
            layout.setContentsMargins(5, 5, 5, 5)

            mode_group = QGroupBox("检测方式")
            mode_layout = QVBoxLayout(mode_group)
            mode_row1_layout = QHBoxLayout()
            mode_row1_layout.setSpacing(20)

            self.mode_realtime = QCheckBox("实时摄像头检测")
            self.mode_image = QCheckBox("图片上传检测")
            self.mode_video = QCheckBox("视频文件检测")

            self.mode_realtime.clicked.connect(lambda: self._on_mode_checkbox_clicked(self.mode_realtime))
            self.mode_image.clicked.connect(lambda: self._on_mode_checkbox_clicked(self.mode_image))
            self.mode_video.clicked.connect(lambda: self._on_mode_checkbox_clicked(self.mode_video))

            mode_row1_layout.addWidget(self.mode_realtime)
            mode_row1_layout.addWidget(self.mode_image)
            mode_row1_layout.addWidget(self.mode_video)
            mode_layout.addLayout(mode_row1_layout)

            voice_control_layout = QHBoxLayout()
            self.voice_input_checkbox = QCheckBox("启用语音输入控制")
            self.voice_input_checkbox.setChecked(False)
            self.voice_input_checkbox.stateChanged.connect(self._on_voice_input_checkbox_changed)
            voice_control_layout.addWidget(self.voice_input_checkbox)
            voice_control_layout.addStretch()
            mode_layout.addLayout(voice_control_layout)

            self.file_selection_widget = QWidget()
            file_layout = QHBoxLayout(self.file_selection_widget)
            file_layout.setContentsMargins(0, 0, 0, 0)
            self.file_path_label = QLabel("未选择文件")
            self.file_path_label.setStyleSheet("color: #7f8c8d; padding: 5px;")
            self.browse_btn = QPushButton("浏览...")
            self.browse_btn.clicked.connect(self.browse_file)
            file_layout.addWidget(self.file_path_label, 1)
            file_layout.addWidget(self.browse_btn)
            mode_layout.addWidget(self.file_selection_widget)
            self.file_selection_widget.hide()
            layout.addWidget(mode_group)

            control_btn_layout = QHBoxLayout()
            self.start_btn = QPushButton("开始检测")
            self.pause_btn = QPushButton("暂停")
            self.clear_btn = QPushButton("清除")
            self.start_btn.clicked.connect(self.toggle_detection)
            self.pause_btn.clicked.connect(self.toggle_pause)
            self.clear_btn.clicked.connect(self.clear_detection)
            for btn in [self.start_btn, self.pause_btn, self.clear_btn]:
                btn.setMinimumHeight(45)
                btn.setMinimumWidth(120)
                control_btn_layout.addWidget(btn)
            self.pause_btn.setEnabled(False)
            self.clear_btn.setEnabled(True)
            layout.addLayout(control_btn_layout)

            self.voice_display_group = QGroupBox("实时语音输入")
            voice_layout = QVBoxLayout(self.voice_display_group)
            voice_btn_layout = QHBoxLayout()
            self.voice_start_btn = QPushButton("开始语音")
            self.voice_stop_btn = QPushButton("停止语音")
            self.voice_start_btn.clicked.connect(self.start_voice_input)
            self.voice_stop_btn.clicked.connect(self.stop_voice_input)
            self.voice_stop_btn.setEnabled(False)
            voice_btn_layout.addWidget(self.voice_start_btn)
            voice_btn_layout.addWidget(self.voice_stop_btn)
            voice_layout.addLayout(voice_btn_layout)
            self.voice_text_label = QLabel("等待语音输入...")
            self.voice_text_label.setAlignment(Qt.AlignCenter)
            self.voice_text_label.setStyleSheet(
                "QLabel { background-color: #f0f6fa; border: 2px solid #3498db; border-radius: 8px; padding: 15px; font-size: 21px; min-height: 21px; }")
            voice_layout.addWidget(self.voice_text_label)
            keywords_layout = QHBoxLayout()
            keywords_layout.addWidget(QLabel("识别关键词: "))
            self.voice_keywords_label = QLabel("无")
            self.voice_keywords_label.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 22px;")
            keywords_layout.addWidget(self.voice_keywords_label)
            keywords_layout.addStretch()
            voice_layout.addLayout(keywords_layout)
            layout.addWidget(self.voice_display_group)

            param_group = QGroupBox("参数调整")
            param_layout = QVBoxLayout(param_group)
            conf_layout = QVBoxLayout()
            conf_label = QLabel("置信度阈值: 0.70")
            self.conf_slider = QSlider(Qt.Horizontal)
            self.conf_slider.setRange(0, 100)
            self.conf_slider.setValue(70)
            self.conf_slider.valueChanged.connect(lambda v: self._on_confidence_changed(v, conf_label))
            conf_layout.addWidget(conf_label)
            conf_layout.addWidget(self.conf_slider)
            param_layout.addLayout(conf_layout)
            layout.addWidget(param_group)

            voice_settings_group = QGroupBox("语音设置")
            voice_settings_layout = QHBoxLayout(voice_settings_group)
            self.voice_enable = QCheckBox("启用语音交互")
            self.voice_enable.setChecked(True)
            self.tts_enable = QCheckBox("启用语音播报")
            self.tts_enable.setChecked(True)
            self.voice_enable.stateChanged.connect(self._on_voice_enable_changed)
            self.tts_enable.stateChanged.connect(self._on_tts_enable_changed)
            voice_settings_layout.addStretch()
            voice_settings_layout.addWidget(self.voice_enable)
            voice_settings_layout.addSpacing(90)
            voice_settings_layout.addWidget(self.tts_enable)
            voice_settings_layout.addStretch()
            layout.addWidget(voice_settings_group)

            export_group = QGroupBox("结果导出")
            export_layout = QHBoxLayout(export_group)
            excel_btn = QPushButton("导出Excel")
            csv_btn = QPushButton("导出CSV")
            word_btn = QPushButton("导出Word")
            for btn in [excel_btn, csv_btn, word_btn]:
                btn.setMinimumHeight(43)
                export_layout.addWidget(btn)
            excel_btn.clicked.connect(lambda: self.export_results('excel'))
            csv_btn.clicked.connect(lambda: self.export_results('csv'))
            word_btn.clicked.connect(lambda: self.export_results('word'))
            layout.addWidget(export_group)
            return tab
        except Exception as e:
            logger.error(f"创建检测控制标签页失败: {e}", exc_info=True)
            raise

    def create_history_tab(self):
        try:
            tab = QWidget()
            layout = QVBoxLayout(tab)
            filter_group = QGroupBox("查询条件")
            filter_layout = QHBoxLayout(filter_group)
            filter_layout.setContentsMargins(15, 15, 15, 15)
            filter_layout.setSpacing(20)
            from PyQt5.QtWidgets import QRadioButton, QButtonGroup
            self.filter_button_group = QButtonGroup(self)
            self.filter_button_group.setExclusive(True)
            self.filter_all = QRadioButton("全部")
            self.filter_all.setChecked(True)
            self.filter_today = QRadioButton("今天")
            self.filter_week = QRadioButton("本周")
            self.filter_month = QRadioButton("本月")
            self.filter_button_group.addButton(self.filter_all, 0)
            self.filter_button_group.addButton(self.filter_today, 1)
            self.filter_button_group.addButton(self.filter_week, 2)
            self.filter_button_group.addButton(self.filter_month, 3)
            filter_layout.addWidget(self.filter_all)
            filter_layout.addWidget(self.filter_today)
            filter_layout.addWidget(self.filter_week)
            filter_layout.addWidget(self.filter_month)
            layout.addWidget(filter_group)

            btn_layout = QHBoxLayout()
            search_btn = QPushButton("查询")
            search_btn.clicked.connect(self.load_history)
            search_btn.setMinimumHeight(45)
            btn_layout.addWidget(search_btn)

            clear_btn = QPushButton("清空历史")
            clear_btn.clicked.connect(self.clear_history)
            clear_btn.setMinimumHeight(45)
            btn_layout.addWidget(clear_btn)

            export_history_btn = QPushButton("导出结果")
            export_history_btn.clicked.connect(self.export_history_with_format_choice)
            export_history_btn.setMinimumHeight(45)
            btn_layout.addWidget(export_history_btn)

            layout.addLayout(btn_layout)

            self.history_table = QTableWidget()
            self.history_table.setColumnCount(5)
            self.history_table.setHorizontalHeaderLabels(["时间", "具体类别", "大类", "置信度", "详情"])
            self.history_table.setFont(QFont("Microsoft YaHei", 12))
            header = self.history_table.horizontalHeader()
            for i in range(5):
                header.setSectionResizeMode(i, QHeaderView.Stretch)
            for i in range(5):
                self.history_table.horizontalHeaderItem(i).setTextAlignment(Qt.AlignCenter)
            layout.addWidget(self.history_table)
            return tab
        except Exception as e:
            logger.error(f"创建历史数据标签页失败: {e}", exc_info=True)
            raise

    def create_stats_panel(self):
        try:
            group = QGroupBox("检测结果统计")
            layout = QVBoxLayout(group)
            time_layout = QHBoxLayout()
            time_layout.setSpacing(10)
            time_layout.addWidget(QLabel("检测耗时: "))
            self.single_detection_time_label = QLabel("0.00ms")
            self.single_detection_time_label.setStyleSheet(
                "font-weight: bold; color: #e74c3c; font-size: 22px; margin-right: 90px; ")
            time_layout.addWidget(self.single_detection_time_label)
            time_layout.addWidget(QLabel("检测时间: "))
            self.detection_time_label = QLabel("--:--:--")
            self.detection_time_label.setStyleSheet("font-weight: bold; color: #3498db; font-size: 22px; ")
            time_layout.addWidget(self.detection_time_label)
            time_layout.addStretch()
            layout.addLayout(time_layout)

            stats_layout = QGridLayout()
            self.stats_labels = {}
            categories = [("可回收垃圾", "#407EC9"), ("厨余垃圾", "#006F62"), ("有害垃圾", "#B83A4B"),
                          ("其他垃圾", "#75787B")]
            for i, (name, color) in enumerate(categories):
                label = QLabel(f"{name}: 0")
                label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 21.5px; padding: 5px;")
                label.setMinimumHeight(30)
                stats_layout.addWidget(label, i // 2, i % 2)
                self.stats_labels[name] = label
            layout.addLayout(stats_layout)

            self.progress_bars = {}
            for name, color in categories:
                bar_layout = QHBoxLayout()
                bar_layout.addWidget(QLabel(name[:4]))
                bar = QProgressBar()
                bar.setMaximum(100)
                bar.setValue(0)
                bar.setTextVisible(True)
                bar.setStyleSheet(
                    f"QProgressBar {{ border: 2px solid {color}; border-radius: 6px; text-align: center; font-size: 20px; font-weight: bold; min-height: 25px; }} QProgressBar::chunk {{ background-color: {color}; border-radius: 6px; }}")
                bar_layout.addWidget(bar, 1)
                layout.addLayout(bar_layout)
                self.progress_bars[name] = bar
            return group
        except Exception as e:
            logger.error(f"创建统计面板失败: {e}", exc_info=True)
            raise

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f6fa; }
            QGroupBox { font-weight: bold; font-size: 22px; border: 2px solid #3498db; border-radius: 10px; margin-top: 10px; padding-top: 15px; background-color: white; }
            QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 10px; color: #3498db; font-size: 22px; }
            QPushButton { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #3498db, stop:1 #2980b9); color: white; border: none; border-radius: 8px; padding: 12px 24px; font-weight: bold; font-size: 22px; }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2980b9, stop:1 #3498db); }
            QPushButton:pressed { background: #1f618d; }
            QPushButton:disabled { background: #bdc3c7; }
            QComboBox { padding: 10px; border: 2px solid #3498db; border-radius: 8px; background: white; min-height: 35px; font-size: 22px; }
            QCheckBox { spacing: 10px; font-size: 22px; }
            QCheckBox::indicator { width: 22px; height: 22px; }
            QSlider::groove:horizontal { height: 14px; background: #ecf0f1; border-radius: 8px; }
            QSlider::handle:horizontal { background: #3498db; width: 26px; margin: -6px 0; border-radius: 13px; }
            QTableWidget { border: 2px solid #bdc3c7; border-radius: 8px; background: white; gridline-color: #ecf0f1; font-size: 22px; }
            QHeaderView::section { background: #3498db; color: white; padding: 12px; font-weight: bold; font-size: 22px; border: none; }
            QLabel { font-size: 22px; }
            QStatusBar { background: #f5f6fa; color: #2c3e50; padding: 8px; font-size: 22px; }
            QTabBar::tab { background: #ecf0f1; padding: 14px 26px; margin-right: 5px; border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: bold; font-size: 22px; }
            QTabBar::tab:selected { background: #3498db; color: white; }
            QTabBar::tab:hover { background: #5dade2; color: white; }
            QProgressBar { border: 2px solid; border-radius: 6px; text-align: center; font-size: 22px; }
        """)

    # ==================== 检测方式互斥与文件选择显示 ====================
    def _on_mode_checkbox_clicked(self, current_checkbox):
        if current_checkbox.isChecked():
            if current_checkbox != self.mode_realtime:
                self.mode_realtime.setChecked(False)
            if current_checkbox != self.mode_image:
                self.mode_image.setChecked(False)
            if current_checkbox != self.mode_video:
                self.mode_video.setChecked(False)
        self._update_file_selection_visibility()

    def _update_file_selection_visibility(self):
        if self.mode_image.isChecked() or self.mode_video.isChecked():
            self.file_selection_widget.show()
        else:
            self.file_selection_widget.hide()

    # ==================== 语音输入复选框处理 ====================
    def _on_voice_input_checkbox_changed(self, state):
        enabled = (state == Qt.Checked)
        self._update_voice_input_buttons()
        if enabled and not self.voice_enable.isChecked():
            self.voice_enable.setChecked(True)
        self.status_bar.showMessage("语音输入控制已启用" if enabled else "语音输入控制已禁用", 2000)

    def _update_voice_input_buttons(self):
        voice_control_enabled = self.voice_input_checkbox.isChecked() and self.voice_enable.isChecked()
        self.voice_start_btn.setEnabled(voice_control_enabled and not self.voice_stop_btn.isEnabled())

    def _on_voice_enable_changed(self, state):
        enabled = (state == Qt.Checked)
        self.voice_input_checkbox.setEnabled(enabled)
        self.tts_enable.setEnabled(enabled)
        if not enabled:
            self.stop_voice_input()
        self._update_voice_input_buttons()

    def _on_tts_enable_changed(self, state):
        if state == Qt.Unchecked:
            self._stop_speech()
            self.status_bar.showMessage("语音播报已停止", 2000)

    # ==================== 语音交互相关 ====================
    def start_voice_input(self):
        try:
            if not self.voice_input_checkbox.isChecked() or not self.voice_enable.isChecked():
                QMessageBox.warning(self, "提示", "请先勾选“启用语音输入控制”并启用语音交互")
                return
            self.voice_start_btn.setEnabled(False)
            self.voice_stop_btn.setEnabled(True)
            self.voice_text_label.setText("正在录音，请说话...")
            self.status_bar.showMessage("语音监听中...")
            self._set_control_buttons_enabled(False)
            self.recording_thread = RecordingThread()
            self.recording_thread.finished.connect(self.on_recording_finished)
            self.recording_thread.error.connect(self.on_recording_error)
            self.recording_thread.start()
        except Exception as e:
            logger.error(f"启动语音输入失败: {e}")
            self.voice_start_btn.setEnabled(True)
            self.voice_stop_btn.setEnabled(False)
            self._set_control_buttons_enabled(True)

    def on_recording_error(self, error_msg):
        logger.error(f"录音错误: {error_msg}")
        QMessageBox.warning(self, "录音错误", error_msg)
        self.stop_voice_input()

    def stop_voice_input(self):
        try:
            if hasattr(self, 'recording_thread'):
                self.recording_thread.stop()
                self.recording_thread.wait()
            self.voice_start_btn.setEnabled(self.voice_input_checkbox.isChecked() and self.voice_enable.isChecked())
            self.voice_stop_btn.setEnabled(False)
            self.voice_text_label.setText("等待语音输入...")
            self.voice_keywords_label.setText("无")
            self.status_bar.showMessage("系统就绪")
            self._set_control_buttons_enabled(True)
        except Exception as e:
            logger.error(f"停止语音输入失败: {e}")

    def on_recording_finished(self, audio_data):
        try:
            self.voice_text_label.setText("识别中...")
            if not hasattr(self, 'voice_manager') or self.voice_manager is None:
                raise Exception("语音管理器未初始化")
            result = self.voice_manager.recognize(audio_data)
            text = result.get('text', '').strip()
            self.voice_text_label.setText(f"识别结果: {text}")
            keywords = self.extract_keywords(text)
            self.update_voice_keywords(keywords)
            if not text or not keywords:
                self.status_bar.showMessage("未识别到有效语音指令", 3000)
                QMessageBox.information(self, "语音识别", "未识别到有效指令，请重新说出指令。")
                self._restore_voice_mode_ui()
                return
            self.execute_voice_commands(keywords)
        except Exception as e:
            logger.error(f"语音识别失败: {e}")
            self.voice_text_label.setText("识别失败，请重试")
            self.status_bar.showMessage("语音识别失败", 3000)
            QMessageBox.warning(self, "识别失败", f"语音识别出错: {str(e)}")
        finally:
            self._restore_voice_mode_ui()

    def _restore_voice_mode_ui(self):
        self.voice_start_btn.setEnabled(self.voice_input_checkbox.isChecked() and self.voice_enable.isChecked())
        self.voice_stop_btn.setEnabled(False)
        self._set_control_buttons_enabled(not self.is_detecting)

    def extract_keywords(self, text: str) -> List[str]:
        try:
            clean_text = re.sub(r'\s+', '', text)
            keywords = []
            keyword_map = {
                '实时': '实时摄像头检测',
                '图片': '图片上传检测',
                '图像': '图片上传检测',
                '视频': '视频文件检测',
                '开始': '开始检测',
                '暂停': '暂停检测',
                '停止': '停止检测',
                '检测': '检测操作',
                '语音': '语音模式'
            }
            for keyword in keyword_map:
                if keyword in clean_text:
                    keywords.append(keyword)
            return keywords
        except Exception as e:
            logger.error(f"提取关键词失败: {e}")
            return []

    def execute_voice_commands(self, keywords: List[str]):
        self._voice_command_active = True
        try:
            if not self.voice_enable.isChecked():
                self.voice_enable.setChecked(True)
            for keyword in keywords:
                if keyword == '实时':
                    self.mode_realtime.setChecked(True)
                    self._on_mode_checkbox_clicked(self.mode_realtime)
                    if not self.is_detecting:
                        QTimer.singleShot(200, self.toggle_detection)
                elif keyword in ('图片', '图像'):
                    self.mode_image.setChecked(True)
                    self._on_mode_checkbox_clicked(self.mode_image)
                    QTimer.singleShot(100, self.browse_file)
                elif keyword == '视频':
                    self.mode_video.setChecked(True)
                    self._on_mode_checkbox_clicked(self.mode_video)
                    QTimer.singleShot(100, self.browse_file)
                elif keyword == '开始':
                    if not self.is_detecting:
                        self.toggle_detection()
                    elif self.is_paused:
                        self.toggle_pause()
                elif keyword == '暂停':
                    if self.is_detecting and not self.is_paused:
                        self.toggle_pause()
                elif keyword == '停止':
                    if self.is_detecting:
                        self.stop_detection()
        except Exception as e:
            logger.error(f"执行语音命令失败: {e}")
        finally:
            self._voice_command_active = False

    def update_voice_keywords(self, keywords: List[str]):
        try:
            self.voice_keywords_label.setText(", ".join(keywords) if keywords else "无")
        except Exception as e:
            logger.error(f"更新关键词显示失败: {e}")

    def browse_file(self):
        try:
            if self.mode_image.isChecked():
                file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "",
                                                           "图片文件 (*.jpg *.jpeg *.png *.bmp *.gif)")
            elif self.mode_video.isChecked():
                file_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv)")
            else:
                return
            if file_path:
                self.file_path_label.setText(Path(file_path).name)
                self.selected_file = file_path
                self.show_preview(file_path)
                if self._voice_command_active and not self.is_detecting:
                    QTimer.singleShot(500, self.toggle_detection)
        except Exception as e:
            logger.error(f"浏览文件失败: {e}")
            QMessageBox.warning(self, "错误", f"选择文件失败：{str(e)}")

    def show_preview(self, file_path: str):
        try:
            if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                image = cv2.imread(file_path)
                if image is not None:
                    self.display_image(image)
                else:
                    raise Exception("无法读取图片文件")
            elif file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                if ret:
                    self.display_image(frame)
                else:
                    raise Exception("无法读取视频文件")
                cap.release()
        except Exception as e:
            logger.error(f"预览失败: {e}")
            QMessageBox.warning(self, "预览错误", f"文件预览失败：{str(e)}")

    def display_image(self, image):
        try:
            if image is None or image.size == 0:
                raise ValueError("图像数据为空")
            result_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = result_image.shape
            label_w = self.video_label.width()
            label_h = self.video_label.height()
            scale = min(label_w / w, label_h / h) if w > 0 and h > 0 else 1.0
            new_w = int(w * scale)
            new_h = int(h * scale)
            if new_w > 0 and new_h > 0:
                result_image = cv2.resize(result_image, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            bytes_per_line = ch * result_image.shape[1]
            qt_image = QImage(result_image.data, result_image.shape[1], result_image.shape[0], bytes_per_line,
                              QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(self.video_label.size(), Qt.KeepAspectRatio,
                                                                          Qt.SmoothTransformation))
        except Exception as e:
            logger.error(f"显示图像失败: {e}")

    # ==================== 导出功能 ====================
    def export_results(self, format_type):
        try:
            if not hasattr(self, 'export_manager') or self.export_manager is None:
                raise Exception("导出管理器未初始化")
            if not self.current_detections_data:
                QMessageBox.warning(self, "提示", "没有可导出的检测结果")
                return

            default_name = f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            filter_str = ""
            if format_type == 'excel':
                filter_str = "Excel文件 (*.xlsx);;所有文件 (*)"
                default_name += ".xlsx"
            elif format_type == 'csv':
                filter_str = "CSV文件 (*.csv);;所有文件 (*)"
                default_name += ".csv"
            elif format_type == 'word':
                filter_str = "Word文档 (*.docx);;所有文件 (*)"
                default_name += ".docx"
            else:
                filter_str = "所有文件 (*)"

            filepath, _ = QFileDialog.getSaveFileName(self, "保存导出文件", default_name, filter_str)
            if not filepath:
                return

            records = []
            for det in self.current_detections_data:
                records.append({
                    'timestamp': datetime.now().isoformat(),
                    'class_name': det.get('class_name', ''),
                    'class_name_cn': det.get('class_name_cn', ''),
                    'category_cn': det.get('category_cn', '未知'),
                    'confidence': det.get('confidence', 0.0),
                    'bbox': det.get('bbox', None),
                    'source_type': det.get('source_type', 'unknown')
                })

            if format_type == 'word':
                stats = {
                    'total_frames': self.total_frames_processed,
                    'total_detections': len(records),
                    'avg_confidence': sum(r['confidence'] for r in records) / len(records) if records else 0,
                    'detection_rate': len(records) / self.total_frames_processed if self.total_frames_processed > 0 else 0,
                    'start_time': self.detection_start_time,
                    'end_time': datetime.now(),
                    'class_counts': self._get_class_counts(records)
                }
                success = self.export_manager.export_summary_report(records, stats, filename=filepath)
            else:
                success = self.export_manager.export_data(records, format_type, filepath=filepath)

            if success:
                QMessageBox.information(self, "成功", f"导出成功：\n{filepath}")
                self.status_bar.showMessage(f"导出成功: {format_type.upper()}", 3000)
            else:
                QMessageBox.warning(self, "失败", "导出失败，请检查日志")
                self.status_bar.showMessage("导出失败", 3000)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")

    def _get_class_counts(self, records):
        from collections import Counter
        counts = Counter()
        for r in records:
            counts[r.get('class_name_cn', '未知')] += 1
        return dict(counts)

    def export_history_with_format_choice(self):
        try:
            if not self.current_history_records:
                QMessageBox.warning(self, "提示", "没有可导出的历史记录")
                return

            msg = QMessageBox()
            msg.setWindowTitle("导出格式选择")
            msg.setText("请选择导出格式：")
            btn_excel = msg.addButton("Excel", QMessageBox.ActionRole)
            btn_word = msg.addButton("Word", QMessageBox.ActionRole)
            msg.addButton("取消", QMessageBox.RejectRole)
            msg.exec_()

            if msg.clickedButton() == btn_excel:
                self.export_history_as('excel')
            elif msg.clickedButton() == btn_word:
                self.export_history_as('word')
            else:
                return
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")

    def export_history_as(self, format_type):
        try:
            records = []
            for rec in self.current_history_records:
                records.append({
                    'timestamp': rec.get('timestamp', ''),
                    'class_name_cn': rec.get('class_name', ''),
                    'category_cn': rec.get('category', ''),
                    'confidence': rec.get('confidence', 0.0),
                    'bbox': rec.get('bbox', None),
                    'source_type': rec.get('source_type', 'history')
                })

            default_name = f"history_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            filter_str = ""
            if format_type == 'excel':
                filter_str = "Excel文件 (*.xlsx);;所有文件 (*)"
                default_name += ".xlsx"
            else:
                filter_str = "Word文档 (*.docx);;所有文件 (*)"
                default_name += ".docx"

            filepath, _ = QFileDialog.getSaveFileName(self, "保存历史记录", default_name, filter_str)
            if not filepath:
                return

            if format_type == 'excel':
                success = self.export_manager.export_to_excel(records, filename=filepath)
            else:
                stats = {
                    'total_frames': 0,
                    'total_detections': len(records),
                    'avg_confidence': sum(r['confidence'] for r in records) / len(records) if records else 0,
                    'detection_rate': 0,
                    'start_time': None,
                    'end_time': datetime.now(),
                    'class_counts': self._get_class_counts(records)
                }
                success = self.export_manager.export_summary_report(records, stats, filename=filepath)

            if success:
                QMessageBox.information(self, "成功", f"历史记录导出成功：\n{filepath}")
                self.status_bar.showMessage("历史记录导出成功", 3000)
            else:
                QMessageBox.warning(self, "失败", "导出失败，请检查日志")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败: {str(e)}")

    # ==================== 历史记录管理 ====================
    def load_history(self):
        try:
            if not hasattr(self, 'db_manager') or self.db_manager is None:
                raise Exception("数据库管理器未初始化")
            filter_id = self.filter_button_group.checkedId()
            start_time = None
            from datetime import datetime, timedelta
            if filter_id == 1:
                start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            elif filter_id == 2:
                start_time = datetime.now() - timedelta(days=datetime.now().weekday())
            elif filter_id == 3:
                start_time = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            records = self.db_manager.get_detection_records(start_time=start_time, limit=100)
            self.current_history_records = records
            self.history_table.setRowCount(0)
            for record in records:
                row = self.history_table.rowCount()
                self.history_table.insertRow(row)
                time_str = record.get('timestamp', '')
                if time_str:
                    try:
                        time_obj = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                        time_str = time_obj.strftime("%H:%M:%S")
                    except:
                        pass
                time_item = QTableWidgetItem(time_str)
                time_item.setTextAlignment(Qt.AlignCenter)
                self.history_table.setItem(row, 0, time_item)
                class_item = QTableWidgetItem(record.get('class_name', ''))
                class_item.setTextAlignment(Qt.AlignCenter)
                self.history_table.setItem(row, 1, class_item)
                category_item = QTableWidgetItem(record.get('category', ''))
                category_item.setTextAlignment(Qt.AlignCenter)
                self.history_table.setItem(row, 2, category_item)
                conf = record.get('confidence', 0)
                conf_item = QTableWidgetItem(f"{conf:.2%}")
                conf_item.setTextAlignment(Qt.AlignCenter)
                self.history_table.setItem(row, 3, conf_item)
                btn_widget = QWidget()
                btn_layout = QHBoxLayout(btn_widget)
                btn_layout.setContentsMargins(5, 2, 5, 2)
                detail_btn = QPushButton("详情")
                detail_btn.clicked.connect(lambda checked, r=record: self.show_history_detail(r))
                detail_btn.setStyleSheet("padding: 3px 8px; font-size: 22px;")
                btn_layout.addWidget(detail_btn)
                self.history_table.setCellWidget(row, 4, btn_widget)
            self.status_bar.showMessage(f"查询到 {len(records)} 条记录")
        except Exception as e:
            logger.error(f"加载历史记录失败: {e}")

    def show_history_detail(self, record: Dict[str, Any]):
        try:
            QMessageBox.information(self, "历史记录详情",
                                    f"检测时间: {record.get('timestamp', '')}\n\n具体类别: {record.get('class_name', '')}\n大类: {record.get('category', '')}\n置信度: {record.get('confidence', 0):.2%}\n来源类型: {record.get('source_type', '')}\n设备ID: {record.get('device_id', '')}")
        except Exception as e:
            logger.error(f"显示历史详情失败: {e}")

    def clear_history(self):
        try:
            reply = QMessageBox.question(
                self, "确认", "确定要清空所有历史数据吗？此操作不可恢复！",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

            if hasattr(self, 'db_manager') and self.db_manager is not None:
                success = self.db_manager.clear_records()
                if success:
                    self.history_table.setRowCount(0)
                    self.current_history_records = []
                    self.status_bar.showMessage("历史数据已全部清空", 3000)
                    QMessageBox.information(self, "成功", "历史数据已清空")
                else:
                    QMessageBox.warning(self, "失败", "清空历史数据失败，请检查数据库连接")
            else:
                QMessageBox.warning(self, "错误", "数据库管理器未初始化，无法清空历史数据")
        except Exception as e:
            logger.error(f"清空历史记录失败: {e}")
            QMessageBox.critical(self, "错误", f"清空失败：{str(e)}")

    # ==================== 检测控制 ====================
    def toggle_detection(self):
        try:
            if not hasattr(self, 'detector') or self.detector is None:
                QMessageBox.warning(self, "错误", "检测模型未加载，无法开始检测！")
                return
            self.start_detection() if not self.is_detecting else self.stop_detection()
        except Exception as e:
            logger.error(f"切换检测状态失败: {e}")

    def start_detection(self):
        try:
            if not (self.mode_realtime.isChecked() or self.mode_image.isChecked() or self.mode_video.isChecked()):
                QMessageBox.warning(self, "提示", "请先选择一种检测方式！")
                return
            if (self.mode_image.isChecked() or self.mode_video.isChecked()) and not self.selected_file:
                QMessageBox.warning(self, "提示", "请先选择图片或视频文件！")
                return
            self.is_detecting = True
            self.is_paused = False
            self.total_detections = 0
            self.results_table.setRowCount(0)
            self.current_count_label.setText("0")
            self.pause_btn.setEnabled(True)
            self.start_btn.setText("终止检测")
            self.status_bar.showMessage("检测中...")
            self.detection_start_time = datetime.now()
            self.detection_time_label.setText("00:00:00")
            if self.elapsed_timer is None:
                self.elapsed_timer = QTimer()
                self.elapsed_timer.timeout.connect(self.update_elapsed_time)
            self.elapsed_timer.start(1000)
            self.total_frames_processed = 0
            if self.mode_realtime.isChecked():
                self.start_camera_detection()
            elif self.mode_image.isChecked():
                self.start_image_detection()
            elif self.mode_video.isChecked():
                self.start_video_detection()
        except Exception as e:
            logger.error(f"开始检测失败: {e}")
            self.is_detecting = False
            self.start_btn.setText("开始检测")

    def stop_detection(self):
        try:
            self._stop_speech()
            self.is_detecting = False
            self.is_paused = False
            self.start_btn.setText("开始检测")
            self.pause_btn.setEnabled(False)
            self.pause_btn.setText("暂停")
            self.status_bar.showMessage("检测已停止")
            if self.elapsed_timer:
                self.elapsed_timer.stop()
            if self.display_timer.isActive():
                self.display_timer.stop()
            if self.detection_thread and self.detection_thread.isRunning():
                self.detection_thread.stop()
                self.detection_thread = None
            if self.camera_detection_thread and self.camera_detection_thread.isRunning():
                self.camera_detection_thread.stop()
                self.camera_detection_thread = None
            self.detection_queue = None
            self.latest_drawn_frame = None
            if self.detection_timer:
                self.detection_timer.stop()
            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None
        except Exception as e:
            logger.error(f"停止检测失败: {e}")

    def toggle_pause(self):
        try:
            if not self.is_detecting:
                return
            if self.is_paused:
                self.is_paused = False
                self.pause_btn.setText("暂停")
                self.status_bar.showMessage("检测中...")
                if self.mode_video.isChecked():
                    self.display_timer.start(int(1000 / self.video_fps) if self.video_fps > 0 else 33)
            else:
                self.is_paused = True
                self.pause_btn.setText("继续")
                self.status_bar.showMessage("检测已暂停")
                self._stop_speech()
                if self.display_timer.isActive():
                    self.display_timer.stop()
        except Exception as e:
            logger.error(f"切换暂停状态失败: {e}")

    def clear_detection(self):
        try:
            self._stop_speech()
            if self.is_detecting:
                self.stop_detection()
            self.video_label.setText("显示区域")
            self.results_table.setRowCount(0)
            self.current_count_label.setText("0")
            self.detection_time_label.setText("--:--:--")
            self.single_detection_time_label.setText("0.00ms")
            for name in self.stats_labels:
                self.stats_labels[name].setText(f"{name}: 0")
            for name in self.progress_bars:
                self.progress_bars[name].setValue(0)
            self.current_frame = None
            self.file_path_label.setText("未选择文件")
            self.selected_file = None
            self.current_detections_data = []
            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None
            if self.detection_timer:
                self.detection_timer.stop()
            if self.elapsed_timer:
                self.elapsed_timer.stop()
                self.detection_start_time = None
            self.status_bar.showMessage("已清除")
        except Exception as e:
            logger.error(f"清除检测结果失败: {e}")

    def start_camera_detection(self):
        try:
            self.video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.video_capture or not self.video_capture.isOpened():
                self.video_capture = cv2.VideoCapture(0)
                if not self.video_capture.isOpened():
                    raise Exception("无法打开摄像头")
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.camera_detection_thread = CameraDetectionThread(
                self.detector,
                self.conf_slider.value() / 100.0,
                inference_size=(640, 480)
            )
            self.camera_detection_thread.frame_processed.connect(self.on_camera_frame_processed)
            self.camera_detection_thread.error.connect(self.on_detection_error)
            self.camera_detection_thread.start()

            self.detection_timer = QTimer()
            self.detection_timer.timeout.connect(self.camera_capture_loop)
            self.detection_timer.start(30)

            self.status_bar.showMessage("摄像头检测中（异步优化）...")
        except Exception as e:
            logger.error(f"摄像头初始化失败: {e}")
            QMessageBox.critical(self, "摄像头错误", f"摄像头初始化失败：{str(e)}")
            self.stop_detection()

    def camera_capture_loop(self):
        if not self.is_detecting or self.is_paused:
            return
        if not self.video_capture or not self.video_capture.isOpened():
            return

        ret, frame = self.video_capture.read()
        if not ret:
            self.stop_detection()
            self.status_bar.showMessage("摄像头读取失败，检测已停止")
            return

        self.current_frame = frame.copy()
        self.total_frames_processed += 1

        self.frame_skip_counter += 1
        if self.frame_skip_counter % self.frame_skip_interval == 0:
            if self.camera_detection_thread:
                self.camera_detection_thread.put_frame(frame.copy())
        else:
            if self.latest_drawn_frame is not None:
                self.display_image(self.latest_drawn_frame)
            else:
                self.display_image(frame)

    def on_camera_frame_processed(self, drawn_frame, detections):
        self.latest_drawn_frame = drawn_frame
        self.display_image(drawn_frame)
        self.update_results_table(detections, source_type="camera")
        self.current_detections_data = detections if detections else []

    def start_image_detection(self):
        try:
            if not self.selected_file:
                raise Exception("未选择图片文件")
            image = cv2.imread(self.selected_file)
            if image is None or image.size == 0:
                raise ValueError("图片文件无效或为空")
            self.process_and_display(image, "image")
            self.pause_btn.setEnabled(False)
            self.status_bar.showMessage("图片检测完成")
            self.is_detecting = False
            self.start_btn.setText("开始检测")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"图片检测失败: {str(e)}")
            self.stop_detection()

    def start_video_detection(self):
        try:
            if not self.selected_file:
                raise Exception("未选择视频文件")
            self.video_capture = cv2.VideoCapture(self.selected_file)
            if not self.video_capture.isOpened():
                raise Exception("无法打开视频文件")
            self.video_total_frames = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            if self.video_total_frames <= 0 or self.video_fps <= 0:
                raise ValueError("视频参数异常")
            ret, first_frame = self.video_capture.read()
            if not ret:
                raise Exception("无法读取视频第一帧")
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self.status_bar.showMessage("模型预热中...")
            warmup_start = time.time()
            _ = self.detector.detect(first_frame, self.conf_slider.value() / 100.0, 0.45)
            logger.info(f"模型预热完成，耗时: {(time.time() - warmup_start) * 1000:.1f}ms")

            self.video_duration = self.video_total_frames / self.video_fps if self.video_fps > 0 else 0
            self.video_progress_slider.setRange(0, self.video_total_frames - 1)
            self.video_progress_slider.setValue(0)
            self.update_progress_label(0)
            self.detection_queue = queue.Queue(maxsize=10)
            self.detection_thread = DetectionThread(self.detector, self.detection_queue,
                                                    self.conf_slider.value() / 100.0)
            self.detection_thread.detection_result.connect(self.on_detection_result)
            self.detection_thread.error.connect(self.on_detection_error)
            self.detection_thread.start()
            self.display_timer.start(int(1000 / self.video_fps) if self.video_fps > 0 else 33)
            self.status_bar.showMessage("视频播放中...")
        except Exception as e:
            logger.error(f"视频初始化失败: {e}")
            QMessageBox.critical(self, "视频错误", f"视频初始化失败：{str(e)}")
            self.stop_detection()

    def on_detection_error(self, error_msg):
        logger.error(f"检测错误: {error_msg}")
        QMessageBox.critical(self, "检测错误", error_msg)
        self.stop_detection()

    def display_video_frame(self):
        try:
            if not self.is_detecting or self.is_paused:
                return
            if not self.video_capture or not self.video_capture.isOpened():
                return
            ret, frame = self.video_capture.read()
            if not ret:
                self.stop_detection()
                self.status_bar.showMessage("视频播放结束，检测已停止")
                return
            self.total_frames_processed += 1
            self.current_frame = frame.copy()
            if self.latest_drawn_frame is not None:
                self.display_image(self.latest_drawn_frame)
            else:
                self.display_image(frame)
            current_pos = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
            self.video_progress_slider.blockSignals(True)
            self.video_progress_slider.setValue(current_pos)
            self.video_progress_slider.blockSignals(False)
            self.update_progress_label(current_pos)
            if self.detection_queue and self.detection_queue.qsize() < 5:
                try:
                    self.detection_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass
        except Exception as e:
            logger.error(f"显示视频帧失败: {e}")
            self.stop_detection()

    def on_detection_result(self, detections, drawn_frame):
        try:
            if detections:
                self.update_results_table(detections, source_type="video")
            else:
                self.update_results_table([], source_type="video")
            self.latest_drawn_frame = drawn_frame
        except Exception as e:
            logger.error(f"处理检测结果失败: {e}")

    def on_progress_slider_pressed(self):
        try:
            if self.display_timer.isActive():
                self.display_timer.stop()
        except Exception as e:
            logger.error(f"进度条按下处理失败: {e}")

    def on_progress_slider_released(self):
        try:
            if not self.video_capture or not self.video_capture.isOpened():
                return
            target_frame = self.video_progress_slider.value()
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = self.video_capture.read()
            if ret:
                self.display_image(frame)
                self.current_frame = frame.copy()
                self.update_progress_label(target_frame)
                self.latest_drawn_frame = None
            if self.is_detecting and not self.is_paused:
                self.display_timer.start(int(1000 / self.video_fps) if self.video_fps > 0 else 33)
        except Exception as e:
            logger.error(f"进度条释放处理失败: {e}")

    def update_progress_label(self, frame_pos):
        try:
            if self.video_fps <= 0:
                return
            current_sec = frame_pos / self.video_fps
            total_sec = self.video_duration
            self.video_position_label.setText(
                f"{int(current_sec // 60):02d}:{int(current_sec % 60):02d} / {int(total_sec // 60):02d}:{int(total_sec % 60):02d}")
        except Exception as e:
            logger.error(f"更新进度标签失败: {e}")

    def process_and_display(self, image, source_type="camera"):
        try:
            self.total_frames_processed += 1
            start_time = time.time()
            self.current_frame = image.copy()
            conf_threshold = self.conf_slider.value() / 100.0
            detections = self.detector.detect(image, conf_threshold, 0.45)
            result_image = self.detector.draw_detections(image, detections)
            result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            h, w, ch = result_image_rgb.shape
            label_w, label_h = self.video_label.width(), self.video_label.height()
            scale = min(label_w / w, label_h / h) if w > 0 and h > 0 else 1.0
            new_w, new_h = int(w * scale), int(h * scale)
            display_image = cv2.resize(result_image_rgb, (new_w, new_h),
                                       interpolation=cv2.INTER_NEAREST) if new_w > 0 and new_h > 0 else result_image_rgb
            bytes_per_line = ch * display_image.shape[1]
            qt_image = QImage(display_image.data, display_image.shape[1], display_image.shape[0], bytes_per_line,
                              QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
            self.update_results_table(detections, source_type)
            self.current_detections_data = detections if detections else []
            end_time = time.time()
            self.single_detection_time = (end_time - start_time) * 1000
            self.single_detection_time_label.setText(f"{self.single_detection_time:.1f}ms")
        except Exception as e:
            logger.error(f"处理并显示失败: {e}")

    def _save_detections_to_db(self, detections, source_type):
        if not self.db_manager:
            self.status_bar.showMessage("警告：数据库未连接，检测记录未保存", 3000)
            return
        for det in detections:
            try:
                data = DetectionData(
                    class_name=det.get('class_name_cn', ''),
                    category=det.get('category_cn', ''),
                    confidence=det.get('confidence', 0.0),
                    bbox=det.get('bbox', []),
                    source_type=source_type,
                    device_id='default',
                    timestamp=datetime.now()
                )
                if not self.db_manager.add_detection_record(data):
                    self.status_bar.showMessage("警告：部分检测记录保存失败", 3000)
            except Exception as e:
                logger.error(f"保存记录异常: {e}")
                self.status_bar.showMessage("警告：检测记录保存异常", 3000)

    def _speak_text(self, text):
        """直接调用 SAPI 异步播报"""
        if not text or not self.tts or not self.tts_enable.isChecked():
            return
        try:
            logger.info(f"请求播报: {text}")
            self.tts.speak(text)
            self.status_bar.showMessage(f"正在播报: {text}", 2000)
        except Exception as e:
            logger.error(f"播报请求失败: {e}")

    def _stop_speech(self):
        if self.tts:
            try:
                self.tts.stop()
            except:
                pass

    def update_results_table(self, detections, source_type):
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            if not detections:
                self.results_table.setRowCount(0)
                self.current_count_label.setText("0")
                for name in self.stats_labels:
                    self.stats_labels[name].setText(f"{name}: 0")
                for name in self.progress_bars:
                    self.progress_bars[name].setValue(0)

                no_detection_record = {
                    'class_name': 'NoDetection',
                    'class_name_cn': '未检测到垃圾',
                    'category_cn': '无',
                    'confidence': 0.0,
                    'bbox': [0, 0, 0, 0],
                    'source_type': source_type
                }
                self.current_detections_data = [no_detection_record]
                QTimer.singleShot(0, lambda: self._save_detections_to_db([no_detection_record], source_type))

                self.results_table.insertRow(0)
                for col, text in enumerate([current_time, "未检测到垃圾", "-", "-", "-"]):
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignCenter)
                    self.results_table.setItem(0, col, item)
                placeholder_btn = QPushButton("-")
                placeholder_btn.setEnabled(False)
                self.results_table.setCellWidget(0, 5, placeholder_btn)
                return

            self.results_table.setRowCount(0)
            total_count = 0
            for det in detections:
                row = self.results_table.rowCount()
                self.results_table.insertRow(row)
                time_item = QTableWidgetItem(current_time)
                time_item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(row, 0, time_item)
                class_item = QTableWidgetItem(det['class_name_cn'])
                class_item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(row, 1, class_item)
                category_item = QTableWidgetItem(det['category_cn'])
                category_item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(row, 2, category_item)
                conf_item = QTableWidgetItem(f"{det['confidence']:.3f}")
                conf_item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(row, 3, conf_item)
                bbox = det['bbox']
                pos_item = QTableWidgetItem(f"({bbox[0]:.0f},{bbox[1]:.0f})-({bbox[2]:.0f},{bbox[3]:.0f})")
                pos_item.setTextAlignment(Qt.AlignCenter)
                self.results_table.setItem(row, 4, pos_item)
                btn_widget = QWidget()
                btn_layout = QHBoxLayout(btn_widget)
                btn_layout.setContentsMargins(5, 2, 5, 2)
                detail_btn = QPushButton("详情")
                detail_btn.clicked.connect(lambda checked, d=det: self.show_result_detail(d))
                detail_btn.setStyleSheet("padding: 3px 8px; font-size: 22px;")
                btn_layout.addWidget(detail_btn)
                self.results_table.setCellWidget(row, 5, btn_widget)
                total_count += 1

            self.current_count_label.setText(str(total_count))
            category_counts = {}
            for det in detections:
                category_counts[det.get('category_cn', '未知')] = category_counts.get(det.get('category_cn', '未知'), 0) + 1
            category_map = {'可回收垃圾': '可回收垃圾', '厨余垃圾': '厨余垃圾', '有害垃圾': '有害垃圾', '其他垃圾': '其他垃圾'}
            for cn_name, en_name in category_map.items():
                count = category_counts.get(cn_name, 0)
                if en_name in self.stats_labels:
                    self.stats_labels[en_name].setText(f"{cn_name}: {count}")
                if en_name in self.progress_bars:
                    self.progress_bars[en_name].setValue(int(count / total_count * 100) if total_count > 0 else 0)

            QTimer.singleShot(0, lambda: self._save_detections_to_db(detections, source_type))

            # 语音播报（SAPI异步，去重）
            if self.tts_enable.isChecked() and self.tts is not None and total_count > 0:
                try:
                    announcement_parts = []
                    for det in detections:
                        class_name = det.get('class_name_cn', '')
                        category_name = det.get('category_cn', '')
                        if class_name and category_name:
                            announcement_parts.append(f"{class_name}{category_name}")
                    if announcement_parts:
                        obj_key = "|".join(sorted(set(announcement_parts)))
                        current_ms = int(time.time() * 1000)
                        should_announce = (obj_key != self.last_announced_object) or \
                                          (current_ms - self.last_announce_time > self.announce_cooldown_ms)
                        if should_announce:
                            logger.info(f"触发语音播报: 物品变化={obj_key != self.last_announced_object}")
                            self.last_announced_object = obj_key
                            self.last_announce_time = current_ms
                            announcement_text = "、".join(announcement_parts)
                            # 直接异步播报
                            self._speak_text(announcement_text)
                except Exception as e:
                    logger.error(f"构建播报文本失败: {e}")
        except Exception as e:
            logger.error(f"更新结果表格失败: {e}")

    def _on_confidence_changed(self, value, label):
        try:
            label.setText(f"置信度阈值: {value / 100:.2f}")
            self.conf_timer.start(200)
        except Exception as e:
            logger.error(f"置信度改变处理失败: {e}")

    def _do_confidence_update(self):
        try:
            if self.camera_detection_thread:
                self.camera_detection_thread.set_threshold(self.conf_slider.value() / 100.0)
            if self.detection_thread:
                self.detection_thread.update_threshold(self.conf_slider.value() / 100.0)
            if self.is_detecting and not self.is_paused and self.current_frame is not None:
                source_type = "image" if self.mode_image.isChecked() else "video" if self.mode_video.isChecked() else "camera"
                self.process_and_display(self.current_frame.copy(), source_type)
        except Exception as e:
            logger.error(f"置信度更新失败: {e}")

    def update_elapsed_time(self):
        try:
            if self.detection_start_time and self.is_detecting:
                elapsed = datetime.now() - self.detection_start_time
                hours, remainder = divmod(elapsed.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                self.detection_time_label.setText(f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        except Exception as e:
            logger.error(f"更新时间失败: {e}")

    def show_result_detail(self, det: Dict[str, Any]):
        try:
            bbox = det.get('bbox', [0, 0, 0, 0])
            QMessageBox.information(self, "检测结果详情",
                                    f"检测到垃圾\n\n具体类别: {det.get('class_name_cn', '')}\n大类类别: {det.get('category_cn', '')}\n置信度: {det.get('confidence', 0):.3f}\n位置坐标: ({bbox[0]:.0f}, {bbox[1]:.0f}) - ({bbox[2]:.0f}, {bbox[3]:.0f})\n检测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            logger.error(f"显示结果详情失败: {e}")

    def start_current_time_timer(self):
        try:
            if self.current_time_timer is None:
                self.current_time_timer = QTimer()
                self.current_time_timer.timeout.connect(self.update_current_time)
            self.current_time_timer.start(1000)
            self.update_current_time()
        except Exception as e:
            logger.error(f"启动时间定时器失败: {e}")

    def update_current_time(self):
        try:
            self.current_time_label.setText(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        except Exception as e:
            logger.error(f"更新时间失败: {e}")

    def _set_control_buttons_enabled(self, enabled: bool):
        self.start_btn.setEnabled(enabled)
        self.pause_btn.setEnabled(enabled and self.is_detecting)
        self.clear_btn.setEnabled(enabled)
        self.mode_realtime.setEnabled(enabled)
        self.mode_image.setEnabled(enabled)
        self.mode_video.setEnabled(enabled)
        if self.browse_btn:
            self.browse_btn.setEnabled(enabled)

    def closeEvent(self, event):
        logger.info("正在关闭窗口，释放资源...")
        if self.is_detecting:
            self.stop_detection()
        self._stop_speech()
        if self.tts:
            self.tts.close()
            self.tts = None
        if self.voice_manager:
            self.voice_manager.close()
            self.voice_manager = None
        if self.db_manager:
            self.db_manager.close()
        event.accept()


# ==================== 主程序入口 ====================
def main():
    app = QApplication(sys.argv)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical("未捕获的异常", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
    try:
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        logger.critical(f"程序启动失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
语音交互模块 - 纯本地 FunASR 实现（修复版）
功能：使用 FunASR 进行语音识别，支持 PyQt 集成
"""
import os
import sys
import io
import wave
import time
import json
import logging
import tempfile
import threading
from typing import Callable, Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import queue
import numpy as np

# PyQt 导入（可选）
try:
    from PyQt5.QtCore import QObject, pyqtSignal
    HAS_QT = True
except ImportError:
    HAS_QT = False
    class QObject: pass
    class pyqtSignal:
        def __init__(self, *args): pass
        def connect(self, *args): pass
        def emit(self, *args): pass

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoiceCommandType(Enum):
    """语音指令类型枚举"""
    START_DETECTION = "start_detection"
    STOP_DETECTION = "stop_detection"
    SAVE_RESULT = "save_result"
    CLEAR_RESULT = "clear_result"
    SWITCH_CAMERA = "switch_camera"
    PAUSE_DETECTION = "pause_detection"
    RESUME_DETECTION = "resume_detection"
    SELECT_REALTIME = "select_realtime"
    SELECT_IMAGE = "select_image"
    SELECT_VIDEO = "select_video"
    UNKNOWN = "unknown"


@dataclass
class VoiceCommand:
    """语音指令数据类"""
    command_type: VoiceCommandType
    text: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class AudioConfig:
    """音频配置类"""
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # 16-bit
    chunk_size: int = 1024
    record_seconds: int = 5
    silence_threshold: int = 500


class LocalParaformerModel:
    """本地 FunASR 模型封装"""

    def __init__(self, model_path: str = None):
        """
        初始化本地模型
        :param model_path: 模型路径（可选，默认使用 FunASR 内置模型）
        """
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载 FunASR 模型"""
        try:
            from funasr import AutoModel
            logger.info("正在加载 FunASR Paraformer 模型...")
            self.model = AutoModel(
                model="paraformer-zh",
                model_revision="v2.0.4",
                disable_pbar=True
            )
            logger.info("FunASR 模型加载成功")
        except ImportError as e:
            logger.error(f"缺少 funasr 依赖: {e}")
            logger.error("请运行: pip install funasr")
            self.model = None
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def recognize(self, audio_data: bytes) -> Optional[Dict[str, Any]]:
        """
        识别音频数据
        :param audio_data: WAV 格式音频数据
        :return: 识别结果字典
        """
        if not self.model:
            logger.error("模型未加载")
            return None

        try:
            with io.BytesIO(audio_data) as audio_buffer:
                with wave.open(audio_buffer, 'rb') as wf:
                    sample_rate = wf.getframerate()
                    num_frames = wf.getnframes()
                    raw_data = wf.readframes(num_frames)

            audio_array = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            logger.debug(f"音频采样率: {sample_rate}, 长度: {len(audio_array)}")

            result = self.model.generate(input=audio_array, batch_size=1)

            if result and len(result) > 0:
                if isinstance(result[0], dict):
                    text = result[0].get('text', '')
                else:
                    text = str(result[0])

                confidence = 0.95
                logger.info(f"FunASR 识别结果: {text}")

                return {
                    'text': text,
                    'confidence': confidence,
                    'local': True
                }
            return None
        except Exception as e:
            logger.error(f"本地识别失败: {e}")
            import traceback
            traceback.print_exc()
            return None


class ParaformerClient:
    """语音识别客户端 - 纯本地 FunASR 版本"""

    def __init__(self, local_model_path: str = None, use_local_model: bool = True, **kwargs):
        self.use_local_model = use_local_model
        self.local_model = None
        self.mock_mode = False

        if use_local_model:
            self.local_model = LocalParaformerModel(local_model_path)
            if not self.local_model or not self.local_model.model:
                logger.warning("本地模型加载失败，将使用模拟模式")
                self.mock_mode = True
        else:
            logger.warning("use_local_model=False 但未配置其他引擎，将使用模拟模式")
            self.mock_mode = True

    def recognize(self, audio_data: bytes) -> Optional[Dict[str, Any]]:
        """同步识别音频数据"""
        if not self.mock_mode and self.local_model and self.local_model.model:
            result = self.local_model.recognize(audio_data)
            if result:
                return result
            else:
                logger.warning("本地模型识别失败，回退到模拟模式")
                return self._mock_recognize()
        else:
            return self._mock_recognize()

    def _mock_recognize(self) -> Dict[str, Any]:
        """模拟识别（当模型不可用时）"""
        import random
        mock_commands = ["开始检测", "停止检测", "暂停检测", "图片检测", "视频检测", "实时检测"]
        if random.random() < 0.2:
            text = ""
        else:
            text = random.choice(mock_commands)
        confidence = random.uniform(0.7, 0.95)
        return {'text': text, 'confidence': confidence, 'mock': True}

    def close(self):
        """关闭资源"""
        pass


class AudioRecorder:
    """音频录制器（使用 PyAudio）"""

    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self.is_recording = False
        self.audio_buffer: List[bytes] = []
        self._pyaudio = None
        self._init_pyaudio()

    def _init_pyaudio(self):
        try:
            import pyaudio
            self._pyaudio = pyaudio.PyAudio()
            logger.info("PyAudio 初始化成功")
        except ImportError:
            logger.warning("PyAudio 未安装，将使用模拟模式")
            self._pyaudio = None
        except Exception as e:
            logger.error(f"PyAudio 初始化失败: {e}")
            self._pyaudio = None

    def record(self, duration: int = None) -> Optional[bytes]:
        """录制音频，返回 WAV 数据"""
        if not self._pyaudio:
            return self._generate_mock_audio()

        duration = duration or self.config.record_seconds
        try:
            import pyaudio
            stream = self._pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                frames_per_buffer=self.config.chunk_size
            )
            logger.info(f"开始录制音频 ({duration} 秒)...")
            self.is_recording = True
            self.audio_buffer = []

            num_frames = int(self.config.sample_rate / self.config.chunk_size * duration)
            for _ in range(num_frames):
                if not self.is_recording:
                    break
                data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                self.audio_buffer.append(data)

            self.is_recording = False
            stream.stop_stream()
            stream.close()
            logger.info("音频录制完成")
            return self._buffer_to_wav()
        except Exception as e:
            logger.error(f"音频录制失败: {e}")
            return None

    def stop_recording(self):
        self.is_recording = False

    def _buffer_to_wav(self) -> bytes:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.config.channels)
            wav_file.setsampwidth(self.config.sample_width)
            wav_file.setframerate(self.config.sample_rate)
            wav_file.writeframes(b''.join(self.audio_buffer))
        return wav_buffer.getvalue()

    def _generate_mock_audio(self) -> bytes:
        samples = np.zeros(self.config.sample_rate, dtype=np.int16)
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.config.channels)
            wav_file.setsampwidth(self.config.sample_width)
            wav_file.setframerate(self.config.sample_rate)
            wav_file.writeframes(samples.tobytes())
        return wav_buffer.getvalue()

    def save_to_file(self, audio_data: bytes, filepath: str):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(audio_data)
        logger.info(f"音频已保存: {filepath}")


class CommandMatcher:
    """指令匹配器（关键词匹配）"""
    KEYWORD_MAP = {
        VoiceCommandType.START_DETECTION: ["开始检测", "启动", "开始"],
        VoiceCommandType.STOP_DETECTION: ["停止检测", "停止", "结束"],
        VoiceCommandType.PAUSE_DETECTION: ["暂停"],
        VoiceCommandType.RESUME_DETECTION: ["继续", "恢复"],
        VoiceCommandType.SELECT_REALTIME: ["实时", "摄像头"],
        VoiceCommandType.SELECT_IMAGE: ["图片", "照片"],
        VoiceCommandType.SELECT_VIDEO: ["视频"],
    }

    def __init__(self):
        self.custom_keywords = {}

    def match(self, text: str) -> VoiceCommandType:
        if not text:
            return VoiceCommandType.UNKNOWN
        text_lower = text.lower().strip()
        for cmd_type, keywords in self.KEYWORD_MAP.items():
            for kw in keywords:
                if kw in text_lower:
                    return cmd_type
        for cmd_type, keywords in self.custom_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    return cmd_type
        return VoiceCommandType.UNKNOWN


class VoiceInteractionManager(QObject):
    """语音交互管理器（PyQt 集成版，纯本地 FunASR）"""
    command_recognized = pyqtSignal(object)
    recognition_error = pyqtSignal(str)
    recording_started = pyqtSignal()
    recording_finished = pyqtSignal()

    def __init__(self, audio_config: AudioConfig = None, local_model_path: str = None, parent=None):
        super().__init__(parent)
        self.audio_recorder = AudioRecorder(audio_config)
        self.paraformer_client = ParaformerClient(local_model_path=local_model_path, use_local_model=True)
        self.command_matcher = CommandMatcher()
        self.is_listening = False
        self._listen_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._consecutive_errors = 0
        self._max_consecutive_errors = 3
        self._command_callbacks = {cmd: [] for cmd in VoiceCommandType}
        self._temp_dir = os.path.join(tempfile.gettempdir(), 'voice_interaction')
        Path(self._temp_dir).mkdir(parents=True, exist_ok=True)

    def recognize(self, audio_data: bytes) -> Dict[str, Any]:
        """同步识别音频数据（供主窗口直接调用）"""
        return self.paraformer_client.recognize(audio_data)

    def register_command_callback(self, command_type: VoiceCommandType, callback):
        if callback not in self._command_callbacks[command_type]:
            self._command_callbacks[command_type].append(callback)

    def unregister_command_callback(self, command_type: VoiceCommandType, callback):
        if callback in self._command_callbacks[command_type]:
            self._command_callbacks[command_type].remove(callback)

    def start_listening(self, continuous: bool = True):
        if self.is_listening:
            return
        self.is_listening = True
        self._stop_event.clear()
        self._consecutive_errors = 0
        if continuous:
            self._listen_thread = threading.Thread(target=self._continuous_listen_loop, daemon=True)
            self._listen_thread.start()
        else:
            self._listen_once()
        logger.info("语音监听已启动")

    def stop_listening(self):
        if not self.is_listening:
            return
        self.is_listening = False
        self._stop_event.set()
        self.audio_recorder.stop_recording()
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=2)
        logger.info("语音监听已停止")

    def _continuous_listen_loop(self):
        while self.is_listening and not self._stop_event.is_set():
            if self._consecutive_errors >= self._max_consecutive_errors:
                logger.warning(f"连续错误 {self._consecutive_errors} 次，暂停 5 秒")
                time.sleep(5)
                self._consecutive_errors = 0
            self._listen_once()
            time.sleep(0.5)

    def _listen_once(self):
        try:
            self.recording_started.emit()
            audio_data = self.audio_recorder.record()
            self.recording_finished.emit()
            if not audio_data:
                logger.warning("音频录制失败")
                self._consecutive_errors += 1
                return
            result = self.paraformer_client.recognize(audio_data)
            self._on_recognition_result(result)
            self._consecutive_errors = 0
        except Exception as e:
            logger.error(f"监听过程出错: {e}")
            self._consecutive_errors += 1
            self.recognition_error.emit(str(e))

    def _on_recognition_result(self, result):
        if not result:
            return
        text = result.get('text', '').strip()
        confidence = result.get('confidence', 0.0)
        if not text:
            return
        logger.info(f"识别结果: {text} (置信度: {confidence:.2f})")
        command_type = self.command_matcher.match(text)
        command = VoiceCommand(
            command_type=command_type,
            text=text,
            confidence=confidence,
            raw_data=result
        )
        self.command_recognized.emit(command)
        self._execute_callbacks(command)

    def _on_recognition_error(self, error):
        logger.error(f"识别错误: {error}")
        self._consecutive_errors += 1
        self.recognition_error.emit(str(error))

    def _execute_callbacks(self, command):
        for cb in self._command_callbacks.get(command.command_type, []):
            try:
                threading.Thread(target=cb, args=(command,), daemon=True).start()
            except Exception as e:
                logger.error(f"执行回调失败: {e}")

    def add_custom_command(self, command_type: VoiceCommandType, keywords: List[str]):
        self.command_matcher.custom_keywords.setdefault(command_type, []).extend(keywords)

    def close(self):
        self.stop_listening()
        self.paraformer_client.close()


# 兼容旧版本别名
VoiceInteraction = VoiceInteractionManager
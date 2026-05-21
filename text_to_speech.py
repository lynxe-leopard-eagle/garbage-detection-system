import logging
import threading
import queue
import time
from typing import Optional
from PyQt5.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


class TextToSpeech(QObject):
    speech_started = pyqtSignal()
    speech_finished = pyqtSignal()
    init_failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._speak_queue = queue.Queue()
        self._worker_thread = None
        self._running = False
        self._current_engine = None
        self._stop_requested = False
        self._init_engine_test()
        self._start_worker()

    def _init_engine_test(self):
        """测试 pyttsx3 是否可用"""
        try:
            import pyttsx3
            test_engine = pyttsx3.init()
            test_engine.say("")
            test_engine.runAndWait()
            logger.info("pyttsx3 语音引擎测试成功")
        except Exception as e:
            error_msg = f"pyttsx3 初始化失败: {e}"
            logger.error(error_msg)
            self.init_failed.emit(error_msg)

    def _start_worker(self):
        """启动后台播报线程"""
        if self._worker_thread and self._worker_thread.is_alive():
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("语音播报后台线程已启动")

    def _worker_loop(self):
        """后台线程循环，逐个处理播报任务"""
        while self._running:
            try:
                text = self._speak_queue.get(timeout=0.5)
                if text is None:  # 停止信号
                    break
                self._speak_sync(text)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"播报线程异常: {e}")

    def _speak_sync(self, text: str):
        """在独立线程中同步播放，每个任务新建引擎实例"""
        if not text:
            return
        try:
            import pyttsx3
            engine = pyttsx3.init()
            self._current_engine = engine
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            self.speech_started.emit()
            logger.info(f"开始播报: {text}")
            engine.say(text)
            engine.runAndWait()
            if self._stop_requested:
                engine.stop()
        except Exception as e:
            logger.error(f"播放语音失败: {e}")
        finally:
            self._current_engine = None
            self.speech_finished.emit()

    def speak(self, text: str) -> bool:
        """异步播报，将文本放入队列立即返回"""
        if not text or not text.strip():
            return False
        try:
            self._speak_queue.put(text.strip())
            return True
        except Exception as e:
            logger.error(f"添加播报任务失败: {e}")
            return False

    def stop(self):
        """立即停止当前播报并清空队列"""
        self._stop_requested = True
        # 清空队列
        while not self._speak_queue.empty():
            try:
                self._speak_queue.get_nowait()
            except queue.Empty:
                break
        # 停止当前引擎
        if self._current_engine:
            try:
                self._current_engine.stop()
            except:
                pass
        self._stop_requested = False
        logger.info("语音播报已停止，队列已清空")

    def is_available(self) -> bool:
        try:
            import pyttsx3
            return True
        except ImportError:
            return False

    def close(self):
        """关闭线程并释放资源"""
        self._running = False
        self.stop()
        self._speak_queue.put(None)  # 发送停止信号
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2)
        self._current_engine = None
        logger.info("语音播报模块已关闭")


# 全局单例
_tts_instance: Optional[TextToSpeech] = None
_tts_lock = threading.Lock()


def get_tts() -> TextToSpeech:
    global _tts_instance
    with _tts_lock:
        if _tts_instance is None:
            _tts_instance = TextToSpeech()
        return _tts_instance


def speak(text: str) -> bool:
    return get_tts().speak(text)


def stop_speech():
    get_tts().stop()


def is_tts_available() -> bool:
    return get_tts().is_available()

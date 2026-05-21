import os
import sys
import subprocess
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_and_switch_environment():
    """
    检查当前Python环境，如果缺少PyQt5则尝试切换到conda环境
    """
    try:
        from PyQt5 import QtCore
        logger.info(f"当前Python: {sys.executable}")
        logger.info(f"PyQt5已安装，版本: {QtCore.PYQT_VERSION_STR}")
        return True
    except ImportError:
        logger.warning("当前环境缺少PyQt5，尝试切换到conda环境...")
        
        # 尝试使用conda环境重新运行
        conda_env = "garbage-detection"
        
        # 构建conda run命令
        cmd = [
            "conda", "run", "-n", conda_env,
            "python", __file__
        ]
        
        print(f"\n正在使用conda环境 [{conda_env}] 重新启动...")
        print(f"命令: {' '.join(cmd)}\n")
        
        try:
            # 使用conda环境运行当前脚本
            result = subprocess.run(cmd, check=False)
            sys.exit(result.returncode)
        except FileNotFoundError:
            print("错误: 找不到conda命令")
            print("请确保Anaconda/Miniconda已正确安装并添加到PATH")
            return False
        except Exception as e:
            print(f"切换环境失败: {e}")
            return False


def check_dependencies():
    """检查必要的依赖"""
    missing_deps = []
    
    try:
        import torch
        logger.info(f"PyTorch 版本: {torch.__version__}")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import cv2
        logger.info(f"OpenCV 版本: {cv2.__version__}")
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        from ultralytics import YOLO
        logger.info("Ultralytics (YOLO) 可用")
    except ImportError:
        missing_deps.append("ultralytics")
    
    try:
        import sqlalchemy
        logger.info(f"SQLAlchemy 版本: {sqlalchemy.__version__}")
    except ImportError:
        missing_deps.append("sqlalchemy")
    
    try:
        import pyaudio
        logger.info("PyAudio 可用")
    except ImportError:
        logger.warning("PyAudio 未安装，语音录制功能将不可用")
    
    try:
        import pyttsx3
        logger.info("pyttsx3 可用")
    except ImportError:
        logger.warning("pyttsx3 未安装，语音播报功能将不可用")
    
    try:
        import funasr
        logger.info("FunASR 可用")
    except ImportError:
        logger.warning("FunASR 未安装，将使用模拟语音识别")
    
    try:
        import pandas
        logger.info(f"Pandas 版本: {pandas.__version__}")
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import openpyxl
        logger.info("openpyxl 可用")
    except ImportError:
        logger.warning("openpyxl 未安装，Excel导出功能将不可用")
    
    if missing_deps:
        print(f"\n缺少必要的依赖包: {', '.join(missing_deps)}")
        print("请运行以下命令安装:")
        print(f"pip install {' '.join(missing_deps)}")
        return False
    
    return True


def setup_directories():
    """创建必要的目录"""
    dirs = [
        'exports',
        'reports',
        'models',
        'screenshots',
        'logs'
    ]
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        dir_path.mkdir(exist_ok=True)
        logger.info(f"目录已创建: {dir_path}")


def main():
    """主函数"""

    print("=" * 60)
    print("基于深度学习的生活垃圾检测系统")
    print("=" * 60)
    
    # 检查并切换环境（如果需要）
    if not check_and_switch_environment():
        print("环境检查失败，程序退出")
        return
    
    # 现在可以安全导入PyQt5
    try:
        from PyQt5.QtWidgets import QApplication
        from src.gui.main_window import MainWindow
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保PyQt5已正确安装")
        return
    
    # 检查依赖
    if not check_dependencies():
        logger.error("缺少必要的依赖包，程序退出")
        return
    
    # 创建目录
    setup_directories()
    
    # 创建应用
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle('Fusion')
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 显示欢迎信息
    print("\n系统启动完成！")
    print("=" * 60)
    print("使用说明:")
    print("1. 点击'开始检测'启动实时检测")
    print("2. 使用'浏览'按钮选择图片或视频文件")
    print("3. 调整置信度和IOU阈值优化检测效果")
    print("4. 启用语音交互和语音播报功能")
    print("5. 检测结果自动保存到数据库")
    print("6. 支持导出Excel/CSV/Word报告")
    print("7. 使用'截图'按钮保存当前画面")
    print("8. 语音指令支持: '开始检测', '停止检测', '暂停检测'等")
    print("=" * 60)
    
    # 运行应用
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

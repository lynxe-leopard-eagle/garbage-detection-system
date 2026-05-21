import os
import sys
import time
import logging
import threading
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import numpy as np

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


try:
    import torch
    import cv2
    from ultralytics import YOLO
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False
    logger.warning("ultralytics 未安装，YOLO功能将不可用")


class YOLODetector:
    def __init__(self, model_path: str = None, device: str = 'auto'):
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        self.class_names = []
        self.class_mapping = {}
        
        # 加载垃圾分类映射
        self._load_class_mapping()
        
        # 加载模型
        if HAS_ULTRALYTICS:
            self._load_model()
        else:
            logger.error("YOLO功能不可用，请先安装 ultralytics: pip install ultralytics")
    
    def _get_device(self, device: str) -> str:
        """获取推理设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device
    
    def _load_class_mapping(self):
        """加载垃圾分类映射"""
        try:
            import json
            mapping_file = Path(__file__).parent.parent.parent / 'garbage_datasets.json'
            if mapping_file.exists():
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    mapping_data = json.load(f)
                
                self.class_mapping = {}
                for item in mapping_data:
                    class_id = item['id']
                    self.class_mapping[class_id] = {
                        'name_en': item['name_en'],
                        'name_cn': item['name_cn'],
                        'category_en': item['category_en'],
                        'category_cn': item['category_cn']
                    }
                
                # 按ID排序获取类别名称
                sorted_items = sorted(mapping_data, key=lambda x: x['id'])
                self.class_names = [item['name_en'] for item in sorted_items]
                
                logger.info(f"加载了 {len(self.class_mapping)} 个垃圾分类映射")
            else:
                logger.warning(f"映射文件不存在: {mapping_file}")
                # 使用默认类别
                self.class_names = [f"class_{i}" for i in range(44)]
        except Exception as e:
            logger.error(f"加载分类映射失败: {e}")
            self.class_names = [f"class_{i}" for i in range(44)]
    
    def _load_model(self):
        """加载YOLO模型"""
        try:
            # 如果未指定模型路径，尝试使用best.pt
            if not self.model_path:
                best_model_path = Path(__file__).parent.parent.parent / 'runs-yolov11m' / 'best.pt'
                if best_model_path.exists():
                    self.model_path = str(best_model_path)
                    logger.info(f"自动检测到YOLOv11m模型: {self.model_path}")
                else:
                    logger.warning("未找到YOLOv11m模型，将使用默认YOLOv8n模型")
            
            if self.model_path and os.path.exists(self.model_path):
                logger.info(f"加载模型: {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                # 使用默认的YOLO模型
                logger.info("使用默认YOLOv8n模型")
                self.model = YOLO('yolov8n.pt')
            
            # 设置设备
            if self.device == 'cuda':
                self.model.to('cuda')
                logger.info("使用GPU加速")
            else:
                self.model.to('cpu')
                logger.info("使用CPU推理")
            
            logger.info(f"模型加载成功，使用设备: {self.device}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self.model = None
    
    def detect(self, image: np.ndarray, conf_threshold: float = 0.5, iou_threshold: float = 0.45) -> List[Dict]:
        if self.model is None:
            logger.error("模型未加载")
            return []
        
        try:
            # 执行推理 - 支持YOLOv8, YOLOv11等
            results = self.model(
                image,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
                device=self.device
            )
            
            detections = []
            
            # 解析结果 - 兼容不同YOLO版本
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                # 处理每个检测框
                for i in range(len(boxes)):
                    # 获取边界框
                    bbox = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # 获取置信度
                    confidence = float(boxes.conf[i].cpu().numpy())
                    
                    # 获取类别ID
                    class_id = int(boxes.cls[i].cpu().numpy())
                    
                    # 获取类别信息 - 根据训练数据调整类别ID映射
                    # YOLOv11m模型可能需要从0开始索引，而映射文件是0-43
                    if class_id < len(self.class_mapping):
                        class_info = self.class_mapping[class_id]
                    else:
                        # 如果类别ID超出范围，回退到未知类别
                        class_info = {
                            'name_en': f'unknown_{class_id}',
                            'name_cn': f'未知_{class_id}',
                            'category_en': 'Unknown',
                            'category_cn': '未知'
                        }
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_info['name_en'],
                        'class_name_cn': class_info['name_cn'],
                        'category': class_info['category_en'],
                        'category_cn': class_info['category_cn']
                    }
                    
                    detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"检测失败: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
        except ImportError:
            logger.error("需要安装Pillow库: pip install Pillow")
            return image

        # 转换为PIL格式
        result_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(result_image)

        # 尝试加载中文字体
        try:
            # Windows系统常见中文字体
            font_path = "C:/Windows/Fonts/simhei.ttf"  # 黑体
            font = ImageFont.truetype(font_path, 16)
            font_small = ImageFont.truetype(font_path, 14)
        except:
            try:
                # 备用字体
                font_path = "C:/Windows/Fonts/msyh.ttf"  # 微软雅黑
                font = ImageFont.truetype(font_path, 16)
                font_small = ImageFont.truetype(font_path, 14)
            except:
                logger.warning("无法加载中文字体，使用默认字体")
                font = ImageFont.load_default()
                font_small = ImageFont.load_default()
        
        # 类别颜色映射
        category_colors = {
            'Recyclables': (64, 126, 201),      # 蓝色 - 可回收垃圾
            'KitchenWaste': (0, 111, 98),   # 绿色 - 厨余垃圾
            'HazardousWaste': (184, 58, 75),   # 红色 - 有害垃圾
            'OtherGarbage': (117, 120, 123), # 灰色 - 其他垃圾
        }
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = bbox
            
            # 获取类别颜色
            category = det.get('category', 'Unknown')
            color = category_colors.get(category, (255, 255, 255))
            
            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # 准备标签文本
            label = f"{det['class_name_cn']} {det['confidence']:.2f}"
            category_label = f"{det['category_cn']}"
            
            # 计算文本尺寸
            label_bbox = draw.textbbox((0, 0), label, font=font)
            label_width = label_bbox[2] - label_bbox[0]
            category_bbox = draw.textbbox((0, 0), category_label, font=font_small)
            category_width = category_bbox[2] - category_bbox[0]
            
            max_width = max(label_width, category_width)
            
            # 绘制标签背景
            y_offset = y1 - 35 if y1 > 35 else y1 + 35
            draw.rectangle(
                [x1, y_offset, x1 + max_width + 10, y_offset + 30],
                fill=color
            )
            
            # 绘制标签文本（使用PIL绘制中文）
            draw.text((x1 + 5, y_offset + 2), label, font=font, fill=(255, 255, 255))
            draw.text((x1 + 5, y_offset + 16), category_label, font=font_small, fill=(255, 255, 255))
        
        # 转换回OpenCV格式
        return cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
    
    def get_detection_summary(self, detections: List[Dict]) -> Dict:
        """
        获取检测结果的统计信息
        
        Args:
            detections: 检测结果列表
            
        Returns:
            统计信息字典
        """
        if not detections:
            return {
                'total': 0,
                'by_category': {},
                'by_class': {}
            }
        
        summary = {
            'total': len(detections),
            'by_category': {},
            'by_class': {}
        }
        
        for det in detections:
            # 按大类统计
            category = det['category_cn']
            if category not in summary['by_category']:
                summary['by_category'][category] = 0
            summary['by_category'][category] += 1
            
            # 按具体类别统计
            class_name = det['class_name_cn']
            if class_name not in summary['by_class']:
                summary['by_class'][class_name] = 0
            summary['by_class'][class_name] += 1
        
        return summary
    
    def is_available(self) -> bool:
        """检查检测器是否可用"""
        return self.model is not None


# 全局检测器实例
_detector_instance = None
_detector_lock = threading.Lock()


def get_detector(model_path: str = None) -> YOLODetector:
    global _detector_instance
    with _detector_lock:
        if _detector_instance is None:
            _detector_instance = YOLODetector(model_path)
        return _detector_instance

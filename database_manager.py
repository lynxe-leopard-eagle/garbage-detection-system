import os
import sqlite3
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)


@dataclass
class DetectionData:
    """检测记录数据类"""
    class_name: str
    category: str  # 新增大类字段
    confidence: float
    bbox: Optional[List[float]] = None
    source_type: str = "unknown"
    device_id: str = "default"
    timestamp: datetime = None


class DatabaseManager:
    """数据库管理器"""

    def __init__(self, use_mysql: bool = False, host: str = "localhost", port: int = 3306,
                 username: str = "root", password: str = "", database: str = "garbage_detection",
                 charset: str = "utf8mb4"):
        """
        初始化数据库管理器
        :param use_mysql: 是否使用MySQL，否则使用SQLite
        :param host: MySQL主机
        :param port: MySQL端口
        :param username: MySQL用户名
        :param password: MySQL密码
        :param database: 数据库名
        :param charset: 字符集
        """
        self.use_mysql = use_mysql
        if use_mysql:
            self._init_mysql(host, port, username, password, database, charset)
        else:
            self._init_sqlite(database)

    def _init_sqlite(self, database: str):
        """初始化SQLite"""
        if not database.endswith('.db'):
            database += '.db'
        self.db_path = database
        self.conn = None
        self._create_tables_sqlite()
        logger.info(f"使用SQLite数据库: {os.path.abspath(self.db_path)}")

    def _init_mysql(self, host, port, username, password, database, charset):
        """初始化MySQL（暂不实现完整逻辑）"""
        try:
            import pymysql
            self.conn = pymysql.connect(
                host=host,
                port=port,
                user=username,
                password=password,
                database=database,
                charset=charset
            )
            self._create_tables_mysql()
            logger.info(f"使用MySQL数据库: {database}@{host}")
        except ImportError:
            logger.error("请安装 pymysql: pip install pymysql")
            raise
        except Exception as e:
            logger.error(f"MySQL连接失败: {e}")
            raise

    def _create_tables_sqlite(self):
        """创建SQLite表结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                class_name TEXT NOT NULL,
                category TEXT,
                confidence REAL NOT NULL,
                bbox TEXT,
                source_type TEXT,
                device_id TEXT,
                timestamp TEXT
            )
        ''')
        # 可选：为旧表添加category列（如果不存在）
        try:
            cursor.execute("ALTER TABLE detections ADD COLUMN category TEXT")
        except sqlite3.OperationalError:
            # 列已存在，忽略
            pass
        conn.commit()
        conn.close()

    def _create_tables_mysql(self):
        """创建MySQL表结构"""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INT AUTO_INCREMENT PRIMARY KEY,
                class_name VARCHAR(255) NOT NULL,
                category VARCHAR(255),
                confidence FLOAT NOT NULL,
                bbox TEXT,
                source_type VARCHAR(50),
                device_id VARCHAR(100),
                timestamp DATETIME
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        ''')
        self.conn.commit()

    def get_connection(self):
        """获取数据库连接（支持重连）"""
        if self.use_mysql:
            if not self.conn or not self.conn.open:
                self._init_mysql(self.host, self.port, self.username, self.password, self.database, self.charset)
            return self.conn
        else:
            # SQLite每次操作新建连接
            return sqlite3.connect(self.db_path)

    def add_detection_record(self, detection: DetectionData) -> bool:
        """
        添加检测记录到数据库
        :param detection: DetectionData对象
        :return: 是否成功
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # 准备数据
            timestamp = detection.timestamp or datetime.now()
            timestamp_str = timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp

            bbox_str = None
            if detection.bbox:
                bbox_str = json.dumps(detection.bbox)

            if self.use_mysql:
                sql = '''
                    INSERT INTO detections 
                    (class_name, category, confidence, bbox, source_type, device_id, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                '''
                cursor.execute(sql, (
                    detection.class_name,
                    detection.category,
                    detection.confidence,
                    bbox_str,
                    detection.source_type,
                    detection.device_id,
                    timestamp_str
                ))
                conn.commit()
            else:
                sql = '''
                    INSERT INTO detections 
                    (class_name, category, confidence, bbox, source_type, device_id, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                '''
                cursor.execute(sql, (
                    detection.class_name,
                    detection.category,
                    detection.confidence,
                    bbox_str,
                    detection.source_type,
                    detection.device_id,
                    timestamp_str
                ))
                conn.commit()
                conn.close()

            logger.debug(f"检测记录已保存: {detection.class_name} ({detection.category})")
            return True
        except Exception as e:
            logger.error(f"保存检测记录失败: {e}")
            return False

    def get_detection_records(self, start_time: datetime = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        查询检测记录
        :param start_time: 起始时间，None表示全部
        :param limit: 最大记录数
        :return: 记录字典列表
        """
        records = []
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            if self.use_mysql:
                if start_time:
                    sql = '''
                        SELECT id, class_name, category, confidence, bbox, source_type, device_id, timestamp
                        FROM detections
                        WHERE timestamp >= %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    '''
                    cursor.execute(sql, (start_time.isoformat(), limit))
                else:
                    sql = '''
                        SELECT id, class_name, category, confidence, bbox, source_type, device_id, timestamp
                        FROM detections
                        ORDER BY timestamp DESC
                        LIMIT %s
                    '''
                    cursor.execute(sql, (limit,))
            else:
                if start_time:
                    sql = '''
                        SELECT id, class_name, category, confidence, bbox, source_type, device_id, timestamp
                        FROM detections
                        WHERE timestamp >= ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    '''
                    cursor.execute(sql, (start_time.isoformat(), limit))
                else:
                    sql = '''
                        SELECT id, class_name, category, confidence, bbox, source_type, device_id, timestamp
                        FROM detections
                        ORDER BY timestamp DESC
                        LIMIT ?
                    '''
                    cursor.execute(sql, (limit,))

            rows = cursor.fetchall()
            for row in rows:
                record = {
                    'id': row[0],
                    'class_name': row[1],
                    'category': row[2],
                    'confidence': row[3],
                    'bbox': row[4],
                    'source_type': row[5],
                    'device_id': row[6],
                    'timestamp': row[7]
                }
                if record['bbox'] and isinstance(record['bbox'], str):
                    try:
                        record['bbox'] = json.loads(record['bbox'])
                    except:
                        pass
                records.append(record)

            if not self.use_mysql:
                conn.close()

            return records
        except Exception as e:
            logger.error(f"查询检测记录失败: {e}")
            return records

    def clear_records(self) -> bool:
        """清空所有检测记录"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            if self.use_mysql:
                cursor.execute("DELETE FROM detections")
                conn.commit()
            else:
                cursor.execute("DELETE FROM detections")
                conn.commit()
                conn.close()
            logger.info("已清空所有检测记录")
            return True
        except Exception as e:
            logger.error(f"清空记录失败: {e}")
            return False

    def close(self):
        """关闭数据库连接"""
        if self.use_mysql and self.conn:
            self.conn.close()
            self.conn = None

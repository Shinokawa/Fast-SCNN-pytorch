#!/usr/bin/env python3
"""
统一日志配置模块 - 确保所有信息都能输出到日志文件

包含：
- setup_unified_logging: 统一日志配置
- get_module_logger: 获取模块专用logger
- LoggerMixin: 为类提供日志功能的混入类
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


class ColoredFormatter(logging.Formatter):
    """带颜色的格式化器（仅在终端输出时生效）"""
    
    COLOR_CODES = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }
    
    def format(self, record):
        # 基础格式化
        formatted = super().format(record)
        
        # 只在stderr输出时添加颜色
        if hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            level_name = record.levelname
            color = self.COLOR_CODES.get(level_name, '')
            reset = self.COLOR_CODES['RESET']
            return f"{color}{formatted}{reset}"
        else:
            return formatted


def setup_unified_logging(log_file: Optional[Union[str, Path]] = None, 
                         level: int = logging.INFO,
                         console_output: bool = True,
                         max_bytes: int = 10 * 1024 * 1024,  # 10MB
                         backup_count: int = 5) -> logging.Logger:
    """
    配置统一的日志系统
    
    参数:
        log_file: 日志文件路径
        level: 日志级别
        console_output: 是否同时输出到控制台
        max_bytes: 日志文件最大大小
        backup_count: 备份文件数量
        
    返回:
        配置好的根logger
    """
    # 清除现有的处理器
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)
    
    # 创建格式化器
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 文件处理器（如果指定了日志文件）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 使用RotatingFileHandler避免日志文件过大
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_path, 
            maxBytes=max_bytes, 
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        print(f"✅ 日志系统已配置 - 文件: {log_path}")
    
    # 控制台处理器
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # 设置第三方库的日志级别，避免过多输出
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    
    return root_logger


def get_module_logger(module_name: str) -> logging.Logger:
    """
    获取模块专用的logger
    
    参数:
        module_name: 模块名称，建议使用 __name__
        
    返回:
        模块专用logger
    """
    return logging.getLogger(module_name)


class LoggerMixin:
    """
    为类提供日志功能的混入类
    
    使用方法:
        class MyClass(LoggerMixin):
            def __init__(self):
                super().__init__()
                self.setup_logger(__name__)
                
            def some_method(self):
                self.logger.info("执行某个操作")
    """
    
    def setup_logger(self, name: str):
        """设置logger"""
        self.logger = get_module_logger(name)
    
    def log_info(self, message: str):
        """记录信息"""
        if hasattr(self, 'logger'):
            self.logger.info(message)
        else:
            print(f"INFO: {message}")
    
    def log_warning(self, message: str):
        """记录警告"""
        if hasattr(self, 'logger'):
            self.logger.warning(message)
        else:
            print(f"WARNING: {message}")
    
    def log_error(self, message: str):
        """记录错误"""
        if hasattr(self, 'logger'):
            self.logger.error(message)
        else:
            print(f"ERROR: {message}")
    
    def log_debug(self, message: str):
        """记录调试信息"""
        if hasattr(self, 'logger'):
            self.logger.debug(message)
        else:
            print(f"DEBUG: {message}")


def log_performance_analysis(times_dict: dict, additional_info: dict = None):
    """
    记录性能分析到日志
    
    参数:
        times_dict: 时间统计字典
        additional_info: 额外信息
    """
    logger = get_module_logger("performance")
    
    total_time = sum(times_dict.values())
    
    logger.info("=" * 60)
    logger.info("🧠 性能分析报告")
    logger.info("=" * 60)
    
    if additional_info:
        for key, value in additional_info.items():
            logger.info(f"📊 {key}: {value}")
        logger.info("-" * 60)
    
    for stage, time_ms in times_dict.items():
        percentage = (time_ms / total_time) * 100
        logger.info(f"⏱️  {stage:15}: {time_ms:6.1f}ms ({percentage:5.1f}%)")
    
    logger.info("-" * 60)
    logger.info(f"🏁 总耗时: {total_time:.1f}ms")
    logger.info(f"⚡ 理论FPS: {1000/total_time:.1f}")
    logger.info("=" * 60)


def log_obstacle_detection_details(obstacles: list, detection_info: dict = None):
    """
    记录障碍物检测详细信息到日志
    
    参数:
        obstacles: 检测到的障碍物列表
        detection_info: 检测过程信息
    """
    logger = get_module_logger("obstacle_detection")
    
    logger.info(f"🚧 障碍物检测完成: 发现 {len(obstacles)} 个障碍物")
    
    if detection_info:
        for key, value in detection_info.items():
            logger.info(f"   {key}: {value}")
    
    for i, obstacle in enumerate(obstacles):
        center_x, center_y = obstacle['center']
        confidence = obstacle['confidence']
        area = obstacle.get('area', 0)
        logger.info(f"   障碍物{i+1}: 中心({center_x:.1f}, {center_y:.1f}), "
                   f"面积{area:.0f}px², 置信度{confidence:.2f}")


def log_system_initialization(module_name: str, config: dict):
    """
    记录系统初始化信息
    
    参数:
        module_name: 模块名称
        config: 配置信息字典
    """
    logger = get_module_logger("initialization")
    
    logger.info(f"🚀 初始化模块: {module_name}")
    for key, value in config.items():
        logger.info(f"   {key}: {value}")
    logger.info(f"✅ {module_name} 初始化完成")


# 便捷函数：用于替换print语句
def log_replace_print(message: str, level: str = "INFO"):
    """
    替换print语句的便捷函数
    
    参数:
        message: 消息内容
        level: 日志级别
    """
    logger = get_module_logger("system")
    
    level = level.upper()
    if level == "DEBUG":
        logger.debug(message)
    elif level == "INFO":
        logger.info(message)
    elif level == "WARNING":
        logger.warning(message)
    elif level == "ERROR":
        logger.error(message)
    elif level == "CRITICAL":
        logger.critical(message)
    else:
        logger.info(message) 
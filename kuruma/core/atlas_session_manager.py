#!/usr/bin/env python3
"""
Atlas NPU 会话管理器

统一管理Atlas环境初始化和多个模型的加载，
支持单进程多线程多Stream架构。
"""

import threading
import time
import numpy as np
from typing import Dict, Optional, Any
from core.logging_config import get_module_logger

# 导入Atlas推理库
try:
    from mindx.sdk import Tensor
    from mindx.sdk import base
    ATLAS_AVAILABLE = True
except ImportError:
    ATLAS_AVAILABLE = False

class AtlasSessionManager:
    """Atlas NPU会话管理器 - 单例模式"""
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(AtlasSessionManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.logger = get_module_logger(__name__)
        self.device_id = 0
        self.models: Dict[str, Any] = {}  # 模型缓存
        self.model_locks: Dict[str, threading.Lock] = {}  # 每个模型的锁
        self._atlas_initialized = False
        self._init_lock = threading.Lock()
        
        AtlasSessionManager._initialized = True
    
    def initialize_atlas(self, device_id: int = 0) -> bool:
        """
        初始化Atlas环境（全局只执行一次）
        
        Args:
            device_id: Atlas设备ID
            
        Returns:
            bool: 初始化是否成功
        """
        if self._atlas_initialized:
            self.logger.info(f"✅ Atlas环境已初始化，设备ID: {self.device_id}")
            return True
        
        with self._init_lock:
            if self._atlas_initialized:
                return True
                
            try:
                if not ATLAS_AVAILABLE:
                    raise ImportError("Atlas NPU环境不可用")
                
                self.logger.info("🚀 初始化Atlas NPU环境...")
                
                # 全局初始化Atlas资源
                base.mx_init()
                self.device_id = device_id
                self._atlas_initialized = True
                
                self.logger.info(f"✅ Atlas NPU环境初始化成功，设备ID: {device_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Atlas环境初始化失败: {e}")
                return False
    
    def load_model(self, model_name: str, model_path: str, device_id: Optional[int] = None) -> Optional[Any]:
        """
        加载模型（支持多个模型并发加载）
        
        Args:
            model_name: 模型名称（用作唯一标识）
            model_path: 模型文件路径
            device_id: 设备ID（可选，默认使用全局设备ID）
            
        Returns:
            模型对象或None
        """
        # 检查模型是否已加载
        if model_name in self.models:
            self.logger.info(f"✅ 模型 {model_name} 已加载，直接返回")
            return self.models[model_name]
        
        # 确保Atlas环境已初始化
        if not self._atlas_initialized:
            if not self.initialize_atlas(device_id or self.device_id):
                return None
        
        # 为每个模型创建独立的锁
        if model_name not in self.model_locks:
            self.model_locks[model_name] = threading.Lock()
        
        with self.model_locks[model_name]:
            # 双重检查锁定
            if model_name in self.models:
                return self.models[model_name]
            
            try:
                self.logger.info(f"📊 加载模型: {model_name} -> {model_path}")
                
                # 加载模型
                target_device_id = device_id if device_id is not None else self.device_id
                model = base.model(modelPath=model_path, deviceId=target_device_id)
                
                # 缓存模型
                self.models[model_name] = model
                
                self.logger.info(f"✅ 模型 {model_name} 加载成功")
                return model
                
            except Exception as e:
                self.logger.error(f"❌ 模型 {model_name} 加载失败: {e}")
                return None
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """
        获取已加载的模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型对象或None
        """
        return self.models.get(model_name)
    
    def unload_model(self, model_name: str):
        """
        卸载指定模型
        
        Args:
            model_name: 模型名称
        """
        if model_name in self.models:
            with self.model_locks.get(model_name, threading.Lock()):
                if model_name in self.models:
                    try:
                        # 这里可以添加模型清理逻辑
                        del self.models[model_name]
                        self.logger.info(f"✅ 模型 {model_name} 已卸载")
                    except Exception as e:
                        self.logger.error(f"❌ 卸载模型 {model_name} 失败: {e}")
    
    def list_models(self) -> list:
        """返回已加载的模型列表"""
        return list(self.models.keys())
    
    def get_device_id(self) -> int:
        """获取当前设备ID"""
        return self.device_id
    
    def is_initialized(self) -> bool:
        """检查Atlas环境是否已初始化"""
        return self._atlas_initialized
    
    def cleanup(self):
        """清理资源"""
        self.logger.info("🧹 清理Atlas会话管理器...")
        
        # 卸载所有模型
        for model_name in list(self.models.keys()):
            self.unload_model(model_name)
        
        # 清理锁
        self.model_locks.clear()
        
        self.logger.info("✅ Atlas会话管理器清理完成")

# 全局单例实例
atlas_session_manager = AtlasSessionManager()

def get_atlas_session() -> AtlasSessionManager:
    """获取Atlas会话管理器实例"""
    return atlas_session_manager

def create_tensor(data) -> Optional[Tensor]:
    """创建Tensor对象的便捷函数"""
    try:
        return Tensor(data)
    except Exception as e:
        logger = get_module_logger(__name__)
        logger.error(f"❌ 创建Tensor失败: {e}")
        return None 
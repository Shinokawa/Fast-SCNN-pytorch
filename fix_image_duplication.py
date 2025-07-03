#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图片去重和优化脚本
用于检查并修复推理脚本中可能的重复处理问题
优化数据加载和推理流程
"""

import os
import sys
import hashlib
import shutil
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image

def calculate_image_hash(image_path):
    """计算图片的MD5哈希值"""
    try:
        with open(image_path, 'rb') as f:
            image_hash = hashlib.md5(f.read()).hexdigest()
        return image_hash
    except Exception as e:
        print(f"Error calculating hash for {image_path}: {e}")
        return None

def calculate_visual_hash(image_path, hash_size=8):
    """计算图片的感知哈希值（用于检测相似图片）"""
    try:
        # 读取图片并转换为灰度
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 缩放到固定大小
        resized = cv2.resize(gray, (hash_size, hash_size))
        
        # 计算平均值
        avg = resized.mean()
        
        # 生成哈希
        hash_bits = []
        for i in range(hash_size):
            for j in range(hash_size):
                hash_bits.append('1' if resized[i, j] > avg else '0')
        
        return ''.join(hash_bits)
    except Exception as e:
        print(f"Error calculating visual hash for {image_path}: {e}")
        return None

def hamming_distance(hash1, hash2):
    """计算两个哈希值之间的汉明距离"""
    if len(hash1) != len(hash2):
        return float('inf')
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

def find_duplicate_images(directory, similarity_threshold=5):
    """查找重复和相似的图片"""
    print(f"扫描目录: {directory}")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
    image_files = []
    
    # 收集所有图片文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(Path(root) / file)
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    # 计算哈希值
    exact_hashes = {}  # MD5哈希 -> [文件路径列表]
    visual_hashes = {}  # 感知哈希 -> [文件路径列表]
    
    for i, image_path in enumerate(image_files):
        print(f"处理 [{i+1}/{len(image_files)}]: {image_path.name}")
        
        # 计算精确哈希（MD5）
        exact_hash = calculate_image_hash(image_path)
        if exact_hash:
            if exact_hash not in exact_hashes:
                exact_hashes[exact_hash] = []
            exact_hashes[exact_hash].append(image_path)
        
        # 计算感知哈希
        visual_hash = calculate_visual_hash(image_path)
        if visual_hash:
            if visual_hash not in visual_hashes:
                visual_hashes[visual_hash] = []
            visual_hashes[visual_hash].append(image_path)
    
    # 查找精确重复
    exact_duplicates = {h: paths for h, paths in exact_hashes.items() if len(paths) > 1}
    
    # 查找相似图片
    similar_groups = []
    processed_hashes = set()
    
    for hash1, paths1 in visual_hashes.items():
        if hash1 in processed_hashes:
            continue
        
        similar_group = list(paths1)
        processed_hashes.add(hash1)
        
        for hash2, paths2 in visual_hashes.items():
            if hash2 in processed_hashes:
                continue
            
            if hamming_distance(hash1, hash2) <= similarity_threshold:
                similar_group.extend(paths2)
                processed_hashes.add(hash2)
        
        if len(similar_group) > 1:
            similar_groups.append(similar_group)
    
    return exact_duplicates, similar_groups

def remove_duplicates(exact_duplicates, backup_dir=None):
    """移除重复图片，保留第一个"""
    removed_count = 0
    
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
    
    for hash_value, paths in exact_duplicates.items():
        print(f"\n发现重复组 (hash: {hash_value[:8]}...):")
        for i, path in enumerate(paths):
            print(f"  {i+1}. {path}")
        
        # 保留第一个文件，移除其他
        keep_file = paths[0]
        remove_files = paths[1:]
        
        print(f"保留: {keep_file}")
        
        for remove_file in remove_files:
            try:
                if backup_dir:
                    # 备份到指定目录
                    backup_path = Path(backup_dir) / f"dup_{remove_file.name}"
                    shutil.move(str(remove_file), str(backup_path))
                    print(f"移动到备份: {remove_file} -> {backup_path}")
                else:
                    # 直接删除
                    os.remove(remove_file)
                    print(f"删除: {remove_file}")
                removed_count += 1
            except Exception as e:
                print(f"删除失败 {remove_file}: {e}")
    
    return removed_count

def optimize_directory_structure(base_dir):
    """优化目录结构，确保没有重复文件"""
    print(f"\n优化目录结构: {base_dir}")
    
    directories_to_check = [
        'data/custom/images',
        'bdd100k/images',
        'static/uploads'
    ]
    
    for rel_dir in directories_to_check:
        full_dir = Path(base_dir) / rel_dir
        if full_dir.exists():
            print(f"\n检查目录: {full_dir}")
            exact_duplicates, similar_groups = find_duplicate_images(full_dir)
            
            if exact_duplicates:
                print(f"发现 {len(exact_duplicates)} 组精确重复")
                backup_dir = full_dir / 'duplicates_backup'
                removed = remove_duplicates(exact_duplicates, backup_dir)
                print(f"移除 {removed} 个重复文件")
            else:
                print("未发现精确重复")
            
            if similar_groups:
                print(f"发现 {len(similar_groups)} 组相似图片:")
                for i, group in enumerate(similar_groups):
                    print(f"  相似组 {i+1}: {len(group)} 个文件")
                    for path in group:
                        print(f"    - {path.name}")
            else:
                print("未发现相似图片")

def check_inference_scripts():
    """检查推理脚本中的潜在重复处理问题"""
    print("\n检查推理脚本...")
    
    scripts_to_check = [
        'test_custom_images.py',
        'test_bdd100k_inference.py', 
        'inference_test.py',
        'test_onnx_inference.py'
    ]
    
    issues_found = []
    
    for script in scripts_to_check:
        if os.path.exists(script):
            print(f"检查脚本: {script}")
            
            with open(script, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查可能的重复处理模式
            issues = []
            
            # 检查是否有重复的文件扩展名处理
            if content.count('image_extensions') > 1:
                issues.append("可能存在重复的文件扩展名定义")
            
            # 检查是否有重复的glob模式
            if content.count('.glob(') > 2:
                issues.append("可能存在重复的文件搜索")
            
            # 检查是否有重复的图片加载
            if content.count('cv2.imread') > 1 and content.count('for') > 1:
                issues.append("可能在循环中重复加载图片")
            
            # 检查是否有重复的预处理
            if content.count('preprocess') > 2:
                issues.append("可能存在重复的预处理调用")
            
            if issues:
                issues_found.append((script, issues))
                print(f"  ⚠️  发现潜在问题:")
                for issue in issues:
                    print(f"    - {issue}")
            else:
                print(f"  ✅  未发现明显问题")
    
    return issues_found

def create_optimized_inference_script():
    """创建优化的推理脚本模板"""
    script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化的图片推理脚本 - 无重复处理版本
避免重复加载、重复预处理等问题
"""

import os
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from collections import OrderedDict

class OptimizedInferenceEngine:
    """优化的推理引擎，避免重复处理"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.processed_files = set()  # 记录已处理的文件，避免重复
        self.image_cache = OrderedDict()  # 图片缓存
        self.max_cache_size = 10  # 最大缓存图片数量
        
    def get_image_files(self, directory, extensions=None):
        """获取图片文件列表，去重处理"""
        if extensions is None:
            extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # 使用set避免重复
        seen_files = set()
        image_files = []
        
        directory = Path(directory)
        for ext in extensions:
            # 搜索文件，避免重复添加
            for pattern in [f'*{ext}', f'*{ext.upper()}']:
                for file_path in directory.glob(pattern):
                    # 使用绝对路径的字符串作为唯一标识
                    file_key = str(file_path.resolve())
                    if file_key not in seen_files:
                        seen_files.add(file_key)
                        image_files.append(file_path)
        
        return sorted(image_files)
    
    def load_image_cached(self, image_path):
        """缓存式图片加载，避免重复读取"""
        image_path = str(image_path)
        
        # 检查缓存
        if image_path in self.image_cache:
            # 移动到最后（LRU）
            self.image_cache.move_to_end(image_path)
            return self.image_cache[image_path]
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        # 添加到缓存
        self.image_cache[image_path] = image
        
        # 限制缓存大小
        if len(self.image_cache) > self.max_cache_size:
            self.image_cache.popitem(last=False)
        
        return image
    
    def preprocess_once(self, image, target_size=(1024, 2048)):
        """单次预处理，避免重复"""
        # 只在需要时进行resize
        if image.shape[:2] != target_size:
            image = cv2.resize(image, target_size)
        
        # 转换为tensor（一次性完成所有转换）
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor
    
    def inference_batch(self, image_paths, batch_size=4):
        """批量推理，提高效率"""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # 批量加载和预处理
            batch_tensors = []
            batch_originals = []
            
            for path in batch_paths:
                if str(path) in self.processed_files:
                    print(f"跳过已处理文件: {path}")
                    continue
                
                try:
                    image = self.load_image_cached(path)
                    tensor = self.preprocess_once(image)
                    batch_tensors.append(tensor)
                    batch_originals.append(image)
                    self.processed_files.add(str(path))
                except Exception as e:
                    print(f"处理失败 {path}: {e}")
                    continue
            
            if not batch_tensors:
                continue
            
            # 批量推理
            try:
                batch_tensor = torch.cat(batch_tensors, dim=0)
                with torch.no_grad():
                    predictions = self.model(batch_tensor)
                
                # 处理结果
                for j, (path, pred, orig) in enumerate(zip(batch_paths, predictions, batch_originals)):
                    results.append({
                        'path': path,
                        'prediction': pred.cpu().numpy(),
                        'original': orig
                    })
                    
            except Exception as e:
                print(f"批量推理失败: {e}")
                continue
        
        return results

def main():
    """主函数 - 演示优化的推理流程"""
    # 这里应该加载实际的模型
    # model = load_your_model()
    # engine = OptimizedInferenceEngine(model)
    
    # 获取图片文件（去重）
    # image_files = engine.get_image_files('data/custom/images')
    # print(f"找到 {len(image_files)} 个唯一图片文件")
    
    # 批量推理
    # results = engine.inference_batch(image_files)
    # print(f"成功处理 {len(results)} 个文件")
    
    print("优化推理脚本模板已生成")

if __name__ == '__main__':
    main()
'''
    
    with open('optimized_inference_template.py', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("已创建优化的推理脚本模板: optimized_inference_template.py")

def main():
    """主函数"""
    print("🔍 图片去重和推理优化工具")
    print("=" * 50)
    
    # 获取当前工作目录
    base_dir = os.getcwd()
    print(f"工作目录: {base_dir}")
    
    # 1. 检查推理脚本
    print("\n📝 检查推理脚本...")
    issues = check_inference_scripts()
    
    if issues:
        print(f"\n⚠️  发现 {len(issues)} 个脚本存在潜在问题:")
        for script, script_issues in issues:
            print(f"\n{script}:")
            for issue in script_issues:
                print(f"  - {issue}")
    else:
        print("\n✅ 所有推理脚本检查正常")
    
    # 2. 优化目录结构
    print("\n📁 优化目录结构...")
    optimize_directory_structure(base_dir)
    
    # 3. 创建优化模板
    print("\n🛠️  创建优化模板...")
    create_optimized_inference_script()
    
    print("\n✅ 优化完成!")
    print("\n建议:")
    print("1. 查看生成的 optimized_inference_template.py 模板")
    print("2. 检查备份目录中的重复文件")
    print("3. 根据发现的问题修改相关推理脚本")

if __name__ == '__main__':
    main()

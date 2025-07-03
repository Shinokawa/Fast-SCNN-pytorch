#!/usr/bin/env python3
"""
简化版Web标注服务器
修复mask合并和保存问题
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import base64
from PIL import Image
import io
import json
from datetime import datetime

app = Flask(__name__)

# 配置
INPUT_DIR = 'data/custom/images'
OUTPUT_DIR = 'data/custom/web_masks'  # 新的输出目录
UPLOAD_DIR = 'static/uploads'

# 确保目录存在
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs('static', exist_ok=True)

@app.route('/')
def index():
    return render_template('simple_annotation.html')

@app.route('/api/images')
def get_images():
    """获取图片列表"""
    images = []
    if os.path.exists(INPUT_DIR):
        for filename in os.listdir(INPUT_DIR):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                mask_filename = filename.rsplit('.', 1)[0] + '_mask.png'
                mask_path = os.path.join(OUTPUT_DIR, mask_filename)
                has_mask = os.path.exists(mask_path)
                
                images.append({
                    'filename': filename,
                    'has_mask': has_mask,
                    'mask_filename': mask_filename if has_mask else None
                })
    
    # 去重
    unique_images = {}
    for img in images:
        if img['filename'] not in unique_images:
            unique_images[img['filename']] = img
    
    result = list(unique_images.values())
    result.sort(key=lambda x: x['filename'])
    
    return jsonify(result)

@app.route('/images/<filename>')
def serve_image(filename):
    """提供图片文件"""
    return send_from_directory(INPUT_DIR, filename)

@app.route('/masks/<filename>')
def serve_mask(filename):
    """提供mask文件"""
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/api/save_mask', methods=['POST'])
def save_mask():
    """保存mask"""
    try:
        data = request.json
        image_filename = data['image_filename']
        mask_data = data['mask_data']
        
        # 解码base64图像数据
        if mask_data.startswith('data:image'):
            mask_data = mask_data.split(',')[1]
        
        mask_bytes = base64.b64decode(mask_data)
        mask_image = Image.open(io.BytesIO(mask_bytes))
        
        # 转换为numpy数组
        mask_array = np.array(mask_image)
        
        # 创建标准的二分类mask
        if len(mask_array.shape) == 3:
            # 检测绿色像素作为可驾驶区域
            green_pixels = (mask_array[:,:,1] > 200) & (mask_array[:,:,0] < 100) & (mask_array[:,:,2] < 100)
        else:
            # 灰度图，检测白色像素
            green_pixels = mask_array > 128
        
        # 创建标准mask：红色表示可驾驶区域
        standard_mask = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
        standard_mask[green_pixels] = [0, 0, 255]  # BGR格式：红色
        
        # 保存mask
        mask_filename = image_filename.rsplit('.', 1)[0] + '_mask.png'
        mask_path = os.path.join(OUTPUT_DIR, mask_filename)
        
        cv2.imwrite(mask_path, standard_mask)
        
        # 保存统计信息
        total_pixels = mask_array.shape[0] * mask_array.shape[1]
        drivable_pixels = np.sum(green_pixels)
        drivable_ratio = drivable_pixels / total_pixels
        
        stats = {
            'image': image_filename,
            'mask_file': mask_filename,
            'total_pixels': int(total_pixels),
            'drivable_pixels': int(drivable_pixels),
            'drivable_ratio': float(drivable_ratio),
            'timestamp': datetime.now().isoformat(),
            'annotation_method': 'web_interface'
        }
        
        stats_path = os.path.join(OUTPUT_DIR, mask_filename.replace('.png', '_stats.json'))
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': f'Mask保存成功: {mask_path}',
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'保存失败: {str(e)}'
        }), 500

@app.route('/api/load_mask/<filename>')
def load_mask(filename):
    """加载已存在的mask"""
    try:
        mask_filename = filename.rsplit('.', 1)[0] + '_mask.png'
        mask_path = os.path.join(OUTPUT_DIR, mask_filename)
        
        if not os.path.exists(mask_path):
            return jsonify({'success': False, 'message': 'Mask不存在'})
        
        # 读取mask
        mask = cv2.imread(mask_path)
        if mask is None:
            return jsonify({'success': False, 'message': '无法读取mask文件'})
        
        # 转换为base64
        _, buffer = cv2.imencode('.png', mask)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'mask_data': f'data:image/png;base64,{mask_base64}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'加载mask失败: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("🚀 启动简化版Web标注服务器...")
    print(f"📁 图片目录: {INPUT_DIR}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("🌐 在浏览器中打开: http://127.0.0.1:5000")
    print("💡 支持多边形和画笔标注，自动保存为标准红色mask格式")
    
    app.run(debug=True, host='127.0.0.1', port=5000)

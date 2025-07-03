#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的Web标注服务器
主要改进：
1. 去除图片重复处理
2. 支持自动区域填色（白线间区域检测）
3. 优化mask保存格式
4. 添加批量处理功能
5. 支持undo/redo历史记录
"""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64
from PIL import Image
import io
import json
from pathlib import Path
import uuid
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Global configuration
app.config['INPUT_DIR'] = 'data/custom/images'
app.config['OUTPUT_DIR'] = 'data/custom/masks'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['TEMP_DIR'] = 'static/temp'

# Ensure directories exist
for dir_path in [app.config['INPUT_DIR'], app.config['OUTPUT_DIR'], 
                app.config['UPLOAD_FOLDER'], app.config['TEMP_DIR'], 'static']:
    os.makedirs(dir_path, exist_ok=True)

# Session storage for image processing state
session_data = {}

class ImageProcessor:
    """图像处理工具类"""
    
    @staticmethod
    def detect_lane_area(image_path, debug=False):
        """检测两条白线间的可驾驶区域"""
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                return None, "Failed to load image"
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 高斯模糊
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 边缘检测
            edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
            
            # 霍夫线变换检测直线
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                  minLineLength=50, maxLineGap=10)
            
            if lines is None:
                return None, "No lines detected"
            
            # 筛选接近垂直的线（车道线）
            lane_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # 计算线段角度
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                # 保留接近垂直的线（角度在60-120度之间）
                if 60 <= angle <= 120:
                    lane_lines.append(line[0])
            
            if len(lane_lines) < 2:
                return None, "Less than 2 lane lines detected"
            
            # 创建mask
            mask = np.zeros(gray.shape, dtype=np.uint8)
            
            # 简单的区域填充：找到最左和最右的线，填充中间区域
            left_x = min([min(x1, x2) for x1, y1, x2, y2 in lane_lines])
            right_x = max([max(x1, x2) for x1, y1, x2, y2 in lane_lines])
            
            # 填充中间区域（简化版本）
            h, w = mask.shape
            for y in range(h//2, h):  # 只填充下半部分（道路区域）
                for x in range(max(0, left_x), min(w, right_x)):
                    mask[y, x] = 255
            
            return mask, "Lane area detected successfully"
            
        except Exception as e:
            return None, f"Error in lane detection: {str(e)}"
    
    @staticmethod
    def flood_fill_area(image, start_point, tolerance=30):
        """基于种子点的区域填充"""
        try:
            h, w = image.shape[:2]
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 创建mask
            mask = np.zeros((h + 2, w + 2), np.uint8)
            
            # 执行漫水填充
            cv2.floodFill(image.copy(), mask, start_point, 255, 
                         loDiff=tolerance, upDiff=tolerance)
            
            # 返回填充区域
            return mask[1:-1, 1:-1]
            
        except Exception as e:
            print(f"Flood fill error: {e}")
            return None

@app.route('/')
def index():
    return render_template('improved_annotation.html')

@app.route('/api/images')
def get_images():
    """获取图片列表，去除重复"""
    images = []
    seen_files = set()  # 防止重复处理
    
    if os.path.exists(app.config['INPUT_DIR']):
        for filename in os.listdir(app.config['INPUT_DIR']):
            if filename.lower() in seen_files:
                continue  # 跳过重复文件
                
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                seen_files.add(filename.lower())
                
                # 检查是否已有mask
                mask_filename = filename.rsplit('.', 1)[0] + '.png'
                mask_path = os.path.join(app.config['OUTPUT_DIR'], mask_filename)
                has_mask = os.path.exists(mask_path)
                
                # 获取文件信息
                file_path = os.path.join(app.config['INPUT_DIR'], filename)
                file_stat = os.stat(file_path)
                
                images.append({
                    'filename': filename,
                    'has_mask': has_mask,
                    'size': file_stat.st_size,
                    'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                })
    
    # 按修改时间排序
    images.sort(key=lambda x: x['modified'], reverse=True)
    
    return jsonify({
        'images': images,
        'total': len(images),
        'completed': len([img for img in images if img['has_mask']])
    })

@app.route('/api/image/<filename>')
def get_image(filename):
    """获取图片文件"""
    try:
        response = send_from_directory(app.config['INPUT_DIR'], filename)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except Exception as e:
        print(f"Error serving image {filename}: {e}")
        return jsonify({'error': 'Image not found'}), 404

@app.route('/api/mask/<filename>')
def get_mask(filename):
    """获取现有mask"""
    mask_filename = filename.rsplit('.', 1)[0] + '.png'
    mask_path = os.path.join(app.config['OUTPUT_DIR'], mask_filename)
    
    if os.path.exists(mask_path):
        return send_file(mask_path)
    else:
        # 返回空mask
        image_path = os.path.join(app.config['INPUT_DIR'], filename)
        if os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                empty_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                _, buffer = cv2.imencode('.png', empty_mask)
                return send_file(
                    io.BytesIO(buffer.tobytes()),
                    mimetype='image/png'
                )
        
        return jsonify({'error': 'Image not found'}), 404

@app.route('/api/auto_detect_lanes', methods=['POST'])
def auto_detect_lanes():
    """自动检测车道区域"""
    data = request.json
    filename = data.get('filename')
    
    if not filename:
        return jsonify({'error': 'Missing filename'}), 400
    
    image_path = os.path.join(app.config['INPUT_DIR'], filename)
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    
    # 执行车道检测
    mask, message = ImageProcessor.detect_lane_area(image_path)
    
    if mask is None:
        return jsonify({'error': message}), 500
    
    # 将mask转换为base64
    _, buffer = cv2.imencode('.png', mask)
    mask_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'success': True,
        'message': message,
        'mask_data': f'data:image/png;base64,{mask_b64}'
    })

@app.route('/api/flood_fill', methods=['POST'])
def flood_fill():
    """区域填充"""
    data = request.json
    filename = data.get('filename')
    start_point = data.get('start_point')  # [x, y]
    tolerance = data.get('tolerance', 30)
    
    if not filename or not start_point:
        return jsonify({'error': 'Missing filename or start point'}), 400
    
    image_path = os.path.join(app.config['INPUT_DIR'], filename)
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        return jsonify({'error': 'Failed to load image'}), 500
    
    # 执行漫水填充
    mask = ImageProcessor.flood_fill_area(img, tuple(start_point), tolerance)
    
    if mask is None:
        return jsonify({'error': 'Flood fill failed'}), 500
    
    # 将mask转换为base64
    _, buffer = cv2.imencode('.png', mask)
    mask_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'success': True,
        'mask_data': f'data:image/png;base64,{mask_b64}'
    })

@app.route('/api/save_mask', methods=['POST'])
def save_mask():
    """保存标注的mask"""
    data = request.json
    filename = data.get('filename')
    mask_data = data.get('mask_data')
    
    if not filename or not mask_data:
        return jsonify({'error': 'Missing filename or mask data'}), 400
    
    try:
        # 解码base64 mask数据
        mask_bytes = base64.b64decode(mask_data.split(',')[1])
        mask_image = Image.open(io.BytesIO(mask_bytes))
        
        # 转换为灰度并标准化到0-255
        mask_array = np.array(mask_image.convert('L'))
        
        # 确保mask只有0和255两个值（二值化）
        mask_array = np.where(mask_array > 127, 255, 0).astype(np.uint8)
        
        # 保存mask
        mask_filename = filename.rsplit('.', 1)[0] + '.png'
        mask_path = os.path.join(app.config['OUTPUT_DIR'], mask_filename)
        cv2.imwrite(mask_path, mask_array)
        
        # 创建备份（带时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"{filename.rsplit('.', 1)[0]}_{timestamp}.png"
        backup_path = os.path.join(app.config['OUTPUT_DIR'], 'backup', backup_filename)
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        cv2.imwrite(backup_path, mask_array)
        
        return jsonify({
            'success': True, 
            'message': f'Mask saved: {mask_filename}',
            'backup': backup_filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/batch_process', methods=['POST'])
def batch_process():
    """批量处理图片"""
    data = request.json
    filenames = data.get('filenames', [])
    operation = data.get('operation')  # 'auto_detect' or 'clear_masks'
    
    if not filenames or not operation:
        return jsonify({'error': 'Missing filenames or operation'}), 400
    
    results = []
    
    for filename in filenames:
        try:
            if operation == 'auto_detect':
                image_path = os.path.join(app.config['INPUT_DIR'], filename)
                if os.path.exists(image_path):
                    mask, message = ImageProcessor.detect_lane_area(image_path)
                    if mask is not None:
                        # 保存自动检测的mask
                        mask_filename = filename.rsplit('.', 1)[0] + '.png'
                        mask_path = os.path.join(app.config['OUTPUT_DIR'], mask_filename)
                        cv2.imwrite(mask_path, mask)
                        results.append({'filename': filename, 'status': 'success', 'message': message})
                    else:
                        results.append({'filename': filename, 'status': 'failed', 'message': message})
                else:
                    results.append({'filename': filename, 'status': 'failed', 'message': 'Image not found'})
            
            elif operation == 'clear_masks':
                mask_filename = filename.rsplit('.', 1)[0] + '.png'
                mask_path = os.path.join(app.config['OUTPUT_DIR'], mask_filename)
                if os.path.exists(mask_path):
                    os.remove(mask_path)
                    results.append({'filename': filename, 'status': 'success', 'message': 'Mask cleared'})
                else:
                    results.append({'filename': filename, 'status': 'success', 'message': 'No mask to clear'})
                    
        except Exception as e:
            results.append({'filename': filename, 'status': 'failed', 'message': str(e)})
    
    return jsonify({
        'success': True,
        'results': results,
        'processed': len([r for r in results if r['status'] == 'success'])
    })

@app.route('/upload', methods=['POST'])
def upload_images():
    """上传新图片进行标注"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    uploaded_files = []
    skipped_files = []
    
    for file in files:
        if file.filename and file.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            filename = file.filename
            file_path = os.path.join(app.config['INPUT_DIR'], filename)
            
            # 检查文件是否已存在
            if os.path.exists(file_path):
                skipped_files.append(filename)
                continue
            
            file.save(file_path)
            uploaded_files.append(filename)
    
    return jsonify({
        'success': True,
        'uploaded_files': uploaded_files,
        'skipped_files': skipped_files,
        'uploaded_count': len(uploaded_files),
        'skipped_count': len(skipped_files)
    })

if __name__ == '__main__':
    # 获取本地IP地址用于网络访问
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"Starting Improved Lane Annotation Server...")
    print(f"Local access: http://localhost:5000")
    print(f"Network access: http://{local_ip}:5000")
    print(f"iPad access: http://{local_ip}:5000")
    print(f"")
    print(f"Place your images in: {app.config['INPUT_DIR']}")
    print(f"Masks will be saved to: {app.config['OUTPUT_DIR']}")
    print(f"Backup masks saved to: {app.config['OUTPUT_DIR']}/backup")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

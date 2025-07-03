from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
import base64
from PIL import Image
import io
import json

app = Flask(__name__)
CORS(app)

# Global configuration
app.config['INPUT_DIR'] = 'data/custom/images'
app.config['OUTPUT_DIR'] = 'data/custom/masks'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure directories exist
os.makedirs(app.config['INPUT_DIR'], exist_ok=True)
os.makedirs(app.config['OUTPUT_DIR'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

@app.route('/')
def index():
    return render_template('annotation.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/api/images')
def get_images():
    """Get list of images to annotate"""
    images = []
    if os.path.exists(app.config['INPUT_DIR']):
        for filename in os.listdir(app.config['INPUT_DIR']):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Check if mask already exists
                mask_filename = filename.rsplit('.', 1)[0] + '.png'
                mask_path = os.path.join(app.config['OUTPUT_DIR'], mask_filename)
                has_mask = os.path.exists(mask_path)
                
                images.append({
                    'filename': filename,
                    'has_mask': has_mask
                })
    
    return jsonify({
        'images': images,
        'total': len(images)
    })

@app.route('/api/image/<filename>')
def get_image(filename):
    """Serve an image file"""
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
    """Get existing mask for an image"""
    mask_filename = filename.rsplit('.', 1)[0] + '.png'
    mask_path = os.path.join(app.config['OUTPUT_DIR'], mask_filename)
    
    if os.path.exists(mask_path):
        return send_file(mask_path)
    else:
        # Return empty mask
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

@app.route('/api/save_mask', methods=['POST'])
def save_mask():
    """Save the annotated mask"""
    data = request.json
    filename = data.get('filename')
    mask_data = data.get('mask_data')
    
    if not filename or not mask_data:
        return jsonify({'error': 'Missing filename or mask data'}), 400
    
    try:
        # Decode base64 mask data
        mask_bytes = base64.b64decode(mask_data.split(',')[1])
        mask_image = Image.open(io.BytesIO(mask_bytes))
        
        # Convert to grayscale and normalize to 0-255
        mask_array = np.array(mask_image.convert('L'))
        
        # Save mask
        mask_filename = filename.rsplit('.', 1)[0] + '.png'
        mask_path = os.path.join(app.config['OUTPUT_DIR'], mask_filename)
        cv2.imwrite(mask_path, mask_array)
        
        return jsonify({'success': True, 'message': f'Mask saved: {mask_filename}'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
def upload_images():
    """Upload new images for annotation"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files')
    uploaded_files = []
    
    for file in files:
        if file.filename and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            filename = file.filename
            file_path = os.path.join(app.config['INPUT_DIR'], filename)
            file.save(file_path)
            uploaded_files.append(filename)
    
    return jsonify({
        'success': True,
        'uploaded_files': uploaded_files,
        'count': len(uploaded_files)
    })

if __name__ == '__main__':
    # Get local IP address for network access
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"Starting Lane Annotation Server...")
    print(f"Local access: http://localhost:5000")
    print(f"Network access: http://{local_ip}:5000")
    print(f"iPad access: http://{local_ip}:5000")
    print(f"")
    print(f"Place your images in: {app.config['INPUT_DIR']}")
    print(f"Masks will be saved to: {app.config['OUTPUT_DIR']}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)

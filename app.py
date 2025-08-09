from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
import uuid

app = Flask(__name__)
CORS(app)

# 配置上传文件夹
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class PerspectiveCorrector:
    def __init__(self):
        pass
    
    def detect_corners_auto(self, image):
        """自动检测图像四个角点"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 按面积排序，取最大的轮廓
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours:
            # 近似轮廓
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 如果轮廓有4个点，可能是矩形
            if len(approx) == 4:
                return self.order_points(approx.reshape(4, 2))
        
        # 如果没有找到合适的轮廓，返回图像四个角
        h, w = image.shape[:2]
        return np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)
    
    def order_points(self, pts):
        """对四个角点进行排序：左上、右上、右下、左下"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # 计算各点坐标之和，左上角点和最小，右下角点和最大
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下
        
        # 计算各点坐标之差，右上角点差值最小，左下角点差值最大
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上
        rect[3] = pts[np.argmax(diff)]  # 左下
        
        return rect
    
    def correct_perspective(self, image, src_points, target_width=800, target_height=600):
        """执行透视矫正"""
        # 确保源点格式正确
        src_points = np.array(src_points, dtype=np.float32)
        
        # 定义目标矩形的四个角点
        dst_points = np.array([
            [0, 0],
            [target_width, 0],
            [target_width, target_height],
            [0, target_height]
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 应用透视变换
        corrected = cv2.warpPerspective(image, matrix, (target_width, target_height))
        
        return corrected

corrector = PerspectiveCorrector()

@app.route('/')
def index():
    return send_file('static/index.html')

@app.route('/api/upload', methods=['POST'])
def upload_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        # 生成唯一文件名
        file_id = str(uuid.uuid4())
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{file_id}.{file_extension}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        # 保存文件
        file.save(filepath)
        
        # 读取图像
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': '无法读取图像文件'}), 400
        
        # 自动检测角点
        corners = corrector.detect_corners_auto(image)
        
        # 将图像转换为base64用于前端显示
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'file_id': file_id,
            'image_base64': image_base64,
            'corners': corners.tolist(),
            'image_size': {'width': image.shape[1], 'height': image.shape[0]}
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/correct', methods=['POST'])
def correct_perspective():
    try:
        data = request.json
        file_id = data.get('file_id')
        corners = data.get('corners')
        target_width = data.get('target_width', 800)
        target_height = data.get('target_height', 600)
        
        if not file_id or not corners:
            return jsonify({'error': '缺少必要参数'}), 400
        
        # 查找原始图像文件
        original_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.startswith(file_id)]
        if not original_files:
            return jsonify({'error': '找不到原始文件'}), 404
        
        original_filepath = os.path.join(UPLOAD_FOLDER, original_files[0])
        image = cv2.imread(original_filepath)
        
        # 执行透视矫正
        corrected_image = corrector.correct_perspective(
            image, corners, target_width, target_height
        )
        
        # 保存矫正后的图像
        output_filename = f"{file_id}_corrected.jpg"
        output_filepath = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_filepath, corrected_image)
        
        # 转换为base64
        _, buffer = cv2.imencode('.jpg', corrected_image)
        corrected_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'corrected_image_base64': corrected_base64,
            'output_file': output_filename
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<filename>')
def download_file(filename):
    try:
        filepath = os.path.join(OUTPUT_FOLDER, filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

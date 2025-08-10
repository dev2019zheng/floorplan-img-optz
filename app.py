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
        """自动检测图像四个角点，优先检测绿色线框"""
        h, w = image.shape[:2]
        
        # 方法1：颜色过滤检测绿色线条
        corners = self.detect_green_frame(image)
        if corners is not None:
            return corners
        
        # 方法2：改进的边缘检测
        corners = self.detect_by_improved_edges(image)
        if corners is not None:
            return corners
        
        # 方法3：轮廓面积过滤
        corners = self.detect_by_contour_area(image)
        if corners is not None:
            return corners
        
        # 如果所有方法都失败，返回图像四个角
        return np.array([
            [20, 20],
            [w-20, 20],
            [w-20, h-20],
            [20, h-20]
        ], dtype=np.float32)
    
    def detect_green_frame(self, image):
        """专门检测绿色线框"""
        # 转换到HSV色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 定义绿色的HSV范围
        lower_green1 = np.array([40, 40, 40])
        upper_green1 = np.array([80, 255, 255])
        
        # 创建绿色掩膜
        mask = cv2.inRange(hsv, lower_green1, upper_green1)
        
        # 形态学操作，连接断开的线段
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # 按面积排序
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in contours:
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                img_area = image.shape[0] * image.shape[1]
                
                # 面积应该占图像的一定比例
                if area > img_area * 0.1:
                    # 近似轮廓
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) == 4:
                        return self.order_points(approx.reshape(4, 2))
                    elif len(approx) > 4:
                        # 如果检测到多于4个点，尝试进一步简化
                        epsilon = 0.05 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)
                        if len(approx) == 4:
                            return self.order_points(approx.reshape(4, 2))
        
        return None
    
    def detect_by_improved_edges(self, image):
        """改进的边缘检测方法"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 自适应阈值
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
        
        # 边缘检测
        edges = cv2.Canny(thresh, 30, 100, apertureSize=3)
        
        # 形态学操作
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # 按面积排序
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            for contour in contours:
                # 计算轮廓面积
                area = cv2.contourArea(contour)
                img_area = image.shape[0] * image.shape[1]
                
                # 过滤太小的轮廓
                if area > img_area * 0.1:
                    # 近似轮廓
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) == 4:
                        return self.order_points(approx.reshape(4, 2))
        
        return None
    
    def detect_by_contour_area(self, image):
        """基于轮廓面积的检测方法"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 多个阈值尝试
        thresholds = [100, 120, 150, 80]
        
        for thresh_val in thresholds:
            # 边缘检测
            edges = cv2.Canny(blurred, thresh_val//2, thresh_val, apertureSize=3)
            
            # 查找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # 过滤轮廓
                valid_contours = []
                img_area = image.shape[0] * image.shape[1]
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    # 面积应该在合理范围内
                    if img_area * 0.1 < area < img_area * 0.9:
                        # 计算轮廓的矩形度
                        x, y, w, h = cv2.boundingRect(contour)
                        rect_area = w * h
                        extent = area / rect_area
                        
                        # 矩形度应该比较高
                        if extent > 0.7:
                            valid_contours.append((area, contour))
                
                if valid_contours:
                    # 按面积排序，取最大的
                    valid_contours.sort(key=lambda x: x[0], reverse=True)
                    contour = valid_contours[0][1]
                    
                    # 近似轮廓
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) == 4:
                        return self.order_points(approx.reshape(4, 2))
        
        return None
    
    def detect_room_candidates(self, image):
        """检测户型图中的房间候选区域"""
        # 转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 自适应直方图均衡化增强对比度
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 自适应阈值二值化
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # 形态学操作连接断开的线条
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        
        # 填充小的空洞
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        
        # 查找轮廓
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        room_candidates = []
        img_area = h * w
        
        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            
            # 过滤太小或太大的轮廓
            if area < img_area * 0.05 or area > img_area * 0.8:
                continue
            
            # 计算轮廓的矩形度
            x, y, rw, rh = cv2.boundingRect(contour)
            rect_area = rw * rh
            rectangularity = area / rect_area if rect_area > 0 else 0
            
            # 只保留相对矩形的区域
            if rectangularity < 0.3:
                continue
            
            # 多边形近似
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 如果近似后不是4个点，尝试更严格的近似
            if len(approx) != 4:
                epsilon = 0.05 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 仍然不是4个点，尝试用最小外接矩形
            if len(approx) != 4:
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                approx = np.int0(box)
            
            if len(approx) >= 4:
                # 如果有多于4个点，使用前4个或最小外接矩形
                if len(approx) > 4:
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    corners = np.float32(box)
                else:
                    corners = approx.reshape(4, 2).astype(np.float32)
                
                # 计算房间质量分数（面积 + 矩形度）
                quality_score = area * rectangularity
                
                room_candidates.append({
                    'corners': self.order_points(corners),
                    'area': area,
                    'rectangularity': rectangularity,
                    'quality_score': quality_score,
                    'center': (x + rw//2, y + rh//2)
                })
        
        # 按质量分数排序
        room_candidates.sort(key=lambda x: x['quality_score'], reverse=True)
        
        # 返回前5个最佳候选
        return room_candidates[:5]
    
    def detect_corners_auto_with_debug(self, image):
        """带调试信息的角点检测"""
        h, w = image.shape[:2]
        detection_info = {
            'method_used': 'fallback',
            'room_candidates': [],
            'contours_found': 0,
            'debug_message': ''
        }
        
        # 方法1：房间检测
        room_candidates = self.detect_room_candidates(image)
        if room_candidates:
            # 选择最佳房间（第一个，面积最大的）
            best_room = room_candidates[0]
            corners = best_room['corners']
            detection_info.update({
                'method_used': 'room_detection',
                'room_candidates': room_candidates,
                'debug_message': f'检测到{len(room_candidates)}个候选房间'
            })
            return corners, detection_info
        
        # 方法2：改进的边缘检测
        corners = self.detect_by_improved_edges(image)
        if corners is not None:
            detection_info.update({
                'method_used': 'improved_edges',
                'debug_message': '使用改进边缘检测方法'
            })
            return corners, detection_info
        
        # 方法3：轮廓面积过滤
        corners = self.detect_by_contour_area(image)
        if corners is not None:
            detection_info.update({
                'method_used': 'contour_area',
                'debug_message': '使用轮廓面积过滤方法'
            })
            return corners, detection_info
        
        # 如果所有方法都失败，返回图像四个角
        detection_info.update({
            'method_used': 'fallback',
            'debug_message': '使用默认角点，建议手动调整'
        })
        
        corners = np.array([
            [20, 20],
            [w-20, 20],
            [w-20, h-20],
            [20, h-20]
        ], dtype=np.float32)
        
        return corners, detection_info
    
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
        corners, detection_info = corrector.detect_corners_auto_with_debug(image)
        
        # 将图像转换为base64用于前端显示
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'file_id': file_id,
            'image_base64': image_base64,
            'corners': corners.tolist(),
            'image_size': {'width': image.shape[1], 'height': image.shape[0]},
            'detection_info': detection_info
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

#!/usr/bin/env python3
"""
创建真实户型图测试用例
生成包含多个房间的户型图，用于测试房间检测算法
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_floor_plan():
    """创建一个模拟的户型图"""
    # 创建白色背景
    width, height = 800, 600
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # 墙体颜色 (深灰色)
    wall_color = (50, 50, 50)
    line_thickness = 8
    
    # 外墙
    cv2.rectangle(img, (50, 50), (750, 550), wall_color, line_thickness)
    
    # 客厅 (大房间，左侧)
    living_room = [(50, 50), (400, 300)]
    
    # 卧室1 (右上)
    bedroom1 = [(400, 50), (750, 300)]
    cv2.line(img, (400, 50), (400, 300), wall_color, line_thickness)
    
    # 卧室2 (右下)
    bedroom2 = [(400, 300), (750, 550)]
    cv2.line(img, (400, 300), (750, 300), wall_color, line_thickness)
    
    # 厨房 (左下)
    kitchen = [(50, 300), (400, 550)]
    cv2.line(img, (50, 300), (400, 300), wall_color, line_thickness)
    
    # 内部分隔线 (更细一些)
    inner_thickness = 4
    
    # 厨房和客厅之间的分隔
    cv2.line(img, (200, 300), (200, 550), wall_color, inner_thickness)
    
    # 门洞 (白色线段覆盖墙体)
    door_color = (255, 255, 255)
    door_thickness = 12
    
    # 客厅到卧室1的门
    cv2.line(img, (400, 150), (400, 190), door_color, door_thickness)
    
    # 客厅到厨房的门  
    cv2.line(img, (120, 300), (160, 300), door_color, door_thickness)
    
    # 卧室1到卧室2的门
    cv2.line(img, (500, 300), (540, 300), door_color, door_thickness)
    
    # 厨房内部门
    cv2.line(img, (200, 400), (200, 440), door_color, door_thickness)
    
    return img

def add_room_labels(img):
    """添加房间标签"""
    # 转换为PIL图像以便添加文字
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # 尝试使用系统字体
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20)  # Windows中文字体
        except:
            font = ImageFont.load_default()
    
    # 添加房间标签
    labels = [
        ("客厅", (200, 175), (0, 0, 0)),
        ("卧室1", (575, 175), (0, 0, 0)), 
        ("卧室2", (575, 425), (0, 0, 0)),
        ("厨房", (300, 425), (0, 0, 0)),
        ("储物间", (125, 425), (0, 0, 0))
    ]
    
    for text, pos, color in labels:
        draw.text(pos, text, fill=color, font=font)
    
    # 转换回OpenCV格式
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def add_perspective_distortion(img, strength=0.3):
    """添加透视变形"""
    h, w = img.shape[:2]
    
    # 原始四个角点
    src_points = np.float32([
        [0, 0],
        [w, 0], 
        [w, h],
        [0, h]
    ])
    
    # 变形后的角点 (模拟拍照角度)
    offset = int(min(w, h) * strength)
    dst_points = np.float32([
        [offset, offset//2],           # 左上
        [w - offset//2, offset],       # 右上
        [w - offset, h - offset//2],   # 右下
        [offset//2, h - offset]        # 左下
    ])
    
    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 应用透视变换
    distorted = cv2.warpPerspective(img, matrix, (w, h), borderValue=(240, 240, 240))
    
    return distorted

def main():
    """主函数"""
    print("🏠 正在创建户型图测试用例...")
    
    # 创建基础户型图
    floor_plan = create_floor_plan()
    
    # 添加房间标签
    labeled_plan = add_room_labels(floor_plan.copy())
    
    # 创建透视变形版本
    distorted_plan = add_perspective_distortion(labeled_plan.copy(), 0.2)
    
    # 保存图像
    test_images = [
        ("floor_plan_original.png", labeled_plan, "原始户型图"),
        ("floor_plan_distorted.png", distorted_plan, "透视变形户型图"),
    ]
    
    for filename, image, description in test_images:
        cv2.imwrite(filename, image)
        print(f"✅ 已生成: {filename} - {description}")
    
    print(f"""
🎯 测试用例生成完成！

📁 生成的文件：
- floor_plan_original.png: 标准户型图 (用于算法测试)
- floor_plan_distorted.png: 透视变形户型图 (模拟实际拍照)

🏠 户型图包含：
- 客厅 (左侧大房间)
- 卧室1 (右上)
- 卧室2 (右下) 
- 厨房 (左下)
- 储物间 (左下小房间)

🔧 使用方法：
1. 启动应用: python app.py
2. 上传测试图像
3. 观察房间检测结果
4. 选择不同的候选房间
5. 进行透视矫正

💡 测试要点：
- 算法应该能检测到多个房间候选
- 最大的房间(客厅)应该被优先选择
- 可以通过UI切换不同房间
- 每个房间都应该有合理的矩形边界
""")

if __name__ == "__main__":
    main()

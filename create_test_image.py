"""
创建测试用的户型图样例
"""
import cv2
import numpy as np

def create_test_floorplan():
    """创建一个简单的户型图测试样例"""
    # 创建白色背景
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # 首先绘制外墙（黑色）
    cv2.rectangle(img, (50, 50), (750, 550), (0, 0, 0), 3)
    
    # 绘制房间分隔线（黑色）
    # 客厅和厨房分隔
    cv2.line(img, (300, 50), (300, 300), (0, 0, 0), 2)
    # 卧室分隔
    cv2.line(img, (50, 300), (750, 300), (0, 0, 0), 2)
    cv2.line(img, (400, 300), (400, 550), (0, 0, 0), 2)
    
    # 绘制门
    # 入户门
    cv2.rectangle(img, (380, 50), (420, 52), (255, 0, 0), -1)
    # 房间门
    cv2.rectangle(img, (298, 250), (302, 300), (255, 0, 0), -1)
    cv2.rectangle(img, (398, 350), (402, 400), (255, 0, 0), -1)
    
    # 绘制窗户
    cv2.rectangle(img, (100, 48), (200, 52), (0, 255, 0), -1)
    cv2.rectangle(img, (500, 48), (650, 52), (0, 255, 0), -1)
    cv2.rectangle(img, (748, 400), (752, 500), (0, 255, 0), -1)
    
    # 重要：绘制绿色外边框（这是我们要检测的目标）
    green_color = (0, 200, 0)  # 绿色 (BGR格式)
    thickness = 4
    
    # 绘制绿色外边框
    cv2.rectangle(img, (30, 30), (770, 570), green_color, thickness)
    
    # 添加房间标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Living Room', (80, 200), font, 1, (100, 100, 100), 2)
    cv2.putText(img, 'Kitchen', (320, 200), font, 1, (100, 100, 100), 2)
    cv2.putText(img, 'Bedroom 1', (80, 450), font, 1, (100, 100, 100), 2)
    cv2.putText(img, 'Bedroom 2', (450, 450), font, 1, (100, 100, 100), 2)
    
    return img

def create_realistic_floorplan():
    """创建更真实的户型图，包含明显的绿色边框"""
    # 创建稍大的白色背景
    img = np.ones((700, 900, 3), dtype=np.uint8) * 255
    
    # 绿色边框颜色 (BGR格式)
    green_color = (0, 180, 0)
    
    # 绘制主要的绿色外边框
    cv2.rectangle(img, (50, 50), (850, 650), green_color, 5)
    
    # 在绿色边框内绘制户型图内容
    # 外墙（黑色，稍微内缩）
    cv2.rectangle(img, (60, 60), (840, 640), (0, 0, 0), 2)
    
    # 房间分隔线
    cv2.line(img, (300, 60), (300, 350), (0, 0, 0), 2)  # 客厅厨房分隔
    cv2.line(img, (60, 350), (840, 350), (0, 0, 0), 2)   # 主分隔线
    cv2.line(img, (450, 350), (450, 640), (0, 0, 0), 2)  # 卧室分隔
    cv2.line(img, (650, 350), (650, 640), (0, 0, 0), 2)  # 第三个房间
    
    # 内部小房间
    cv2.line(img, (300, 200), (450, 200), (0, 0, 0), 2)
    cv2.line(img, (375, 60), (375, 200), (0, 0, 0), 2)
    
    # 绘制门（红色）
    door_color = (0, 0, 255)
    cv2.rectangle(img, (400, 60), (440, 62), door_color, -1)   # 入户门
    cv2.rectangle(img, (298, 280), (302, 330), door_color, -1) # 客厅门
    cv2.rectangle(img, (448, 400), (452, 450), door_color, -1) # 卧室门
    cv2.rectangle(img, (600, 348), (650, 352), door_color, -1) # 房间门
    
    # 绘制窗户（蓝色）
    window_color = (255, 100, 0)
    cv2.rectangle(img, (120, 58), (220, 62), window_color, -1)
    cv2.rectangle(img, (500, 58), (600, 62), window_color, -1)
    cv2.rectangle(img, (838, 400), (842, 500), window_color, -1)
    cv2.rectangle(img, (200, 638), (300, 642), window_color, -1)
    
    # 添加房间标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Living Room', (80, 250), font, 0.8, (80, 80, 80), 2)
    cv2.putText(img, 'Kitchen', (320, 150), font, 0.7, (80, 80, 80), 2)
    cv2.putText(img, 'Bathroom', (320, 280), font, 0.6, (80, 80, 80), 2)
    cv2.putText(img, 'Bedroom 1', (80, 500), font, 0.8, (80, 80, 80), 2)
    cv2.putText(img, 'Bedroom 2', (470, 500), font, 0.8, (80, 80, 80), 2)
    cv2.putText(img, 'Study', (670, 500), font, 0.7, (80, 80, 80), 2)
    
    return img

def apply_perspective_distortion(img):
    """对图像应用透视变形，模拟拍摄角度"""
    height, width = img.shape[:2]
    
    # 原始四个角点
    src_points = np.float32([
        [0, 0],           # 左上
        [width, 0],       # 右上
        [width, height],  # 右下
        [0, height]       # 左下
    ])
    
    # 变形后的四个角点（模拟透视拍摄）
    dst_points = np.float32([
        [50, 30],         # 左上稍微偏移
        [width-30, 60],   # 右上
        [width-80, height-40],  # 右下
        [80, height-20]   # 左下
    ])
    
    # 计算透视变换矩阵
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 应用透视变换
    distorted = cv2.warpPerspective(img, matrix, (width, height))
    
    return distorted

if __name__ == "__main__":
    print("正在创建测试户型图...")
    
    # 创建标准户型图（带绿色边框）
    floorplan = create_test_floorplan()
    cv2.imwrite('test_floorplan_original.jpg', floorplan)
    print("✓ 已创建标准户型图: test_floorplan_original.jpg")
    
    # 创建透视变形的户型图
    distorted_floorplan = apply_perspective_distortion(floorplan)
    cv2.imwrite('test_floorplan_distorted.jpg', distorted_floorplan)
    print("✓ 已创建透视变形户型图: test_floorplan_distorted.jpg")
    
    # 创建更真实的户型图
    realistic_floorplan = create_realistic_floorplan()
    cv2.imwrite('test_realistic_floorplan.jpg', realistic_floorplan)
    print("✓ 已创建真实户型图: test_realistic_floorplan.jpg")
    
    # 创建真实户型图的透视变形版本
    realistic_distorted = apply_perspective_distortion(realistic_floorplan)
    cv2.imwrite('test_realistic_distorted.jpg', realistic_distorted)
    print("✓ 已创建真实户型图透视变形版: test_realistic_distorted.jpg")
    
    print("\n🧪 测试说明:")
    print("1. test_floorplan_distorted.jpg - 简单户型图透视变形版（绿色边框）")
    print("2. test_realistic_distorted.jpg - 真实户型图透视变形版（绿色边框）")
    print("3. test_floorplan_original.jpg - 标准参考图像")
    print("4. test_realistic_floorplan.jpg - 真实参考图像")
    print("\n📋 使用方法:")
    print("• 上传变形图像到应用中测试绿色边框检测")
    print("• 检查应用是否正确识别绿色线框而非红色线框")
    print("• 验证透视矫正效果")

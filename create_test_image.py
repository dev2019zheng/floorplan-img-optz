"""
创建测试用的户型图样例
"""
import cv2
import numpy as np

def create_test_floorplan():
    """创建一个简单的户型图测试样例"""
    # 创建白色背景
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # 绘制外墙
    cv2.rectangle(img, (50, 50), (750, 550), (0, 0, 0), 3)
    
    # 绘制房间分隔线
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
    
    # 添加房间标签
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Living Room', (80, 200), font, 1, (100, 100, 100), 2)
    cv2.putText(img, 'Kitchen', (320, 200), font, 1, (100, 100, 100), 2)
    cv2.putText(img, 'Bedroom 1', (80, 450), font, 1, (100, 100, 100), 2)
    cv2.putText(img, 'Bedroom 2', (450, 450), font, 1, (100, 100, 100), 2)
    
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
    # 创建标准户型图
    floorplan = create_test_floorplan()
    cv2.imwrite('test_floorplan_original.jpg', floorplan)
    print("已创建标准户型图: test_floorplan_original.jpg")
    
    # 创建透视变形的户型图
    distorted_floorplan = apply_perspective_distortion(floorplan)
    cv2.imwrite('test_floorplan_distorted.jpg', distorted_floorplan)
    print("已创建透视变形户型图: test_floorplan_distorted.jpg")
    
    print("\n测试说明:")
    print("1. test_floorplan_distorted.jpg - 用于测试透视矫正的变形图像")
    print("2. test_floorplan_original.jpg - 标准参考图像")
    print("3. 上传 test_floorplan_distorted.jpg 到应用中进行透视矫正测试")

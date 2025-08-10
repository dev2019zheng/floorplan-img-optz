#!/usr/bin/env python3
"""
创建简单的户型图测试用例
使用PIL创建，不依赖OpenCV
"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_simple_floor_plan():
    """创建一个简单的户型图"""
    # 创建白色背景
    width, height = 800, 600
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # 墙体颜色 (深灰色)
    wall_color = (50, 50, 50)
    line_width = 8
    
    # 外墙
    draw.rectangle([50, 50, 750, 550], outline=wall_color, width=line_width)
    
    # 客厅和卧室1之间的墙
    draw.line([400, 50, 400, 300], fill=wall_color, width=line_width)
    
    # 卧室1和卧室2之间的墙  
    draw.line([400, 300, 750, 300], fill=wall_color, width=line_width)
    
    # 客厅和厨房之间的墙
    draw.line([50, 300, 400, 300], fill=wall_color, width=line_width)
    
    # 厨房内部分隔
    draw.line([200, 300, 200, 550], fill=wall_color, width=4)
    
    # 门洞 (用白色覆盖)
    door_width = 12
    
    # 客厅到卧室1的门
    draw.line([400, 150, 400, 190], fill='white', width=door_width)
    
    # 客厅到厨房的门  
    draw.line([120, 300, 160, 300], fill='white', width=door_width)
    
    # 卧室1到卧室2的门
    draw.line([500, 300, 540, 300], fill='white', width=door_width)
    
    # 厨房内部门
    draw.line([200, 400, 200, 440], fill='white', width=door_width)
    
    return img

def add_room_labels(img):
    """添加房间标签"""
    draw = ImageDraw.Draw(img)
    
    # 尝试使用系统字体
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20)
        except:
            font = ImageFont.load_default()
    
    # 添加房间标签
    labels = [
        ("客厅", (200, 175)),
        ("卧室1", (575, 175)), 
        ("卧室2", (575, 425)),
        ("厨房", (300, 425)),
        ("储物间", (125, 425))
    ]
    
    for text, pos in labels:
        draw.text(pos, text, fill=(0, 0, 0), font=font)
    
    return img

def add_perspective_effect(img, strength=0.2):
    """添加简单的透视效果"""
    w, h = img.size
    
    # 计算变形参数
    offset_x = int(w * strength)
    offset_y = int(h * strength * 0.5)
    
    # 定义透视变换的四个角点
    transform_data = [
        offset_x, offset_y,           # 左上
        w - offset_x//2, offset_y,    # 右上  
        w - offset_x, h - offset_y,   # 右下
        offset_x//2, h - offset_y     # 左下
    ]
    
    # 应用透视变换
    distorted = img.transform(
        (w, h),
        Image.Transform.QUAD,
        transform_data,
        Image.Resampling.BILINEAR,
        fillcolor=(240, 240, 240)
    )
    
    return distorted

def main():
    """主函数"""
    print("🏠 正在创建户型图测试用例...")
    
    # 创建基础户型图
    floor_plan = create_simple_floor_plan()
    
    # 添加房间标签
    labeled_plan = add_room_labels(floor_plan.copy())
    
    # 创建透视变形版本
    distorted_plan = add_perspective_effect(labeled_plan.copy(), 0.15)
    
    # 保存图像
    test_images = [
        ("test_floor_plan.png", labeled_plan, "标准户型图"),
        ("test_floor_plan_distorted.png", distorted_plan, "透视变形户型图"),
    ]
    
    for filename, image, description in test_images:
        image.save(filename)
        print(f"✅ 已生成: {filename} - {description}")
    
    print(f"""
🎯 测试用例生成完成！

📁 生成的文件：
- test_floor_plan.png: 标准户型图
- test_floor_plan_distorted.png: 透视变形户型图

🏠 户型图布局：
┌─────────┬─────────┐
│  客厅   │  卧室1  │
│         │         │
├─────┬───┼─────────┤
│厨房 │储 │  卧室2  │
│     │物 │         │
└─────┴───┴─────────┘

🔧 测试方法：
1. 启动应用: python app.py
2. 上传 test_floor_plan_distorted.png
3. 查看房间检测结果
4. 选择不同的候选房间进行矫正

💡 预期结果：
- 检测到多个房间候选 (客厅、卧室1、卧室2等)
- 最大房间(客厅)被优先选择  
- 可以通过UI切换选择不同房间
- 矫正后得到规整的矩形
""")

if __name__ == "__main__":
    main()

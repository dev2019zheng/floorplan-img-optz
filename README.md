# 户型图透视矫正应用

这是一个专门用于矫正户型图透视变形的Web应用，可以自动检测图像中的四个角点并进行透视矫正，让倾斜拍摄的户型图变得标准化。

## 功能特点

- 🖼️ **智能角点检测**：自动识别户型图的四个角点
- 🎯 **手动调整**：支持手动点击调整角点位置
- ⚙️ **自定义输出**：可设置输出图像的宽度和高度
- 📱 **响应式设计**：支持桌面和移动设备
- 💾 **便捷下载**：一键下载矫正后的图像

## 快速开始

### 环境要求

- Python 3.7+
- pip

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行应用

```bash
python app.py
```

应用将在 `http://localhost:5000` 启动。

## 使用说明

1. **上传图像**：
   - 点击上传区域选择户型图文件
   - 支持拖拽上传
   - 支持 JPG、PNG 等常见格式

2. **检查角点**：
   - 系统会自动检测四个角点
   - 红色圆点表示检测到的角点
   - 可以点击角点附近区域来调整位置

3. **设置输出尺寸**：
   - 在控制面板中设置目标宽度和高度
   - 默认为 800x600 像素

4. **执行矫正**：
   - 点击"开始透视矫正"按钮
   - 等待处理完成

5. **下载结果**：
   - 矫正完成后点击"下载矫正图像"
   - 保存标准化的户型图

## API 接口

### POST /api/upload
上传图像文件并自动检测角点

**请求**：
- 文件：multipart/form-data 中的 `image` 字段

**响应**：
```json
{
  "file_id": "唯一文件ID",
  "image_base64": "图像的base64编码",
  "corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "image_size": {"width": 宽度, "height": 高度}
}
```

### POST /api/correct
执行透视矫正

**请求**：
```json
{
  "file_id": "文件ID",
  "corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "target_width": 800,
  "target_height": 600
}
```

**响应**：
```json
{
  "corrected_image_base64": "矫正后图像的base64编码",
  "output_file": "输出文件名"
}
```

### GET /api/download/{filename}
下载矫正后的图像文件

## 技术架构

### 后端
- **Flask**：Web框架
- **OpenCV**：图像处理和透视矫正
- **NumPy**：数值计算
- **Pillow**：图像格式处理

### 前端
- **HTML5 Canvas**：图像显示和交互
- **原生JavaScript**：用户界面逻辑
- **CSS3**：现代化的响应式样式

### 算法说明

1. **边缘检测**：使用Canny算法检测图像边缘
2. **轮廓分析**：查找并筛选矩形轮廓
3. **角点排序**：按左上、右上、右下、左下顺序排列
4. **透视变换**：使用OpenCV的getPerspectiveTransform计算变换矩阵
5. **图像矫正**：应用透视变换得到标准化图像

## 文件结构

```
img-opt/
├── app.py              # Flask应用主文件
├── requirements.txt    # Python依赖
├── README.md          # 说明文档
├── static/
│   └── index.html     # 前端界面
├── uploads/           # 上传文件存储（自动创建）
└── outputs/           # 输出文件存储（自动创建）
```

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

MIT License

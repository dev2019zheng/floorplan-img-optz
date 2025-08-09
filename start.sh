#!/bin/bash

echo "正在启动户型图透视矫正应用..."
echo

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3环境"
    echo "请先安装Python 3.7或更高版本"
    exit 1
fi

# 安装依赖
echo "正在安装依赖包..."
pip3 install -r requirements.txt

# 启动应用
echo
echo "正在启动应用服务器..."
echo "请在浏览器中访问: http://localhost:5000"
echo
python3 app.py

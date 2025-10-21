#!/bin/bash
# 快速开始脚本 - YOLO K-Fold交叉验证系统

echo "================================"
echo "YOLO K-Fold 交叉验证快速开始"
echo "================================"

# 检查Python环境
echo ""
echo "检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

echo "Python版本: $(python3 --version)"

# 安装依赖
echo ""
echo "正在安装依赖包..."
pip install -r requirements.txt

# 检查数据目录
echo ""
echo "检查数据目录..."
if [ ! -d "./Dhan-Shomadhan/Field Background" ] || [ ! -d "./Dhan-Shomadhan/White Background" ]; then
    echo "警告: 未找到数据目录 'Field Background' 或 'White Background'"
    echo "请确保数据已正确放置在 ./Dhan-Shomadhan/ 目录下"
    echo ""
    echo "期望的目录结构:"
    echo "Dhan-Shomadhan/"
    echo "├── Field Background/"
    echo "│   └── *.jpg"
    echo "└── White Background/"
    echo "    └── *.jpg"
    exit 1
fi

echo "✓ 数据目录检查通过"

# 运行示例
echo ""
echo "================================"
echo "运行示例：仅数据集划分"
echo "================================"
echo ""
echo "命令: python main.py --mode split --random_seed 42"
echo ""
read -p "是否运行? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python main.py --mode split --random_seed 42
    echo ""
    echo "数据集划分完成！"
    echo "结果保存在: ./Dhan-Shomadhan/42/datasets/"
fi

echo ""
echo "================================"
echo "后续步骤"
echo "================================"
echo ""
echo "1. 训练模型:"
echo "   python main.py --mode train --random_seed 42"
echo ""
echo "2. 测试模型:"
echo "   python main.py --mode test --random_seed 42"
echo ""
echo "3. 生成可视化:"
echo "   python main.py --mode visualize --random_seed 42"
echo ""
echo "4. 完整流程:"
echo "   python main.py --mode all --random_seed 42"
echo ""
echo "5. 查看帮助:"
echo "   python main.py --help"
echo ""

# YOLOv8 水稻病害检测系统

基于YOLOv8的智能水稻病害检测系统，能够自动识别和定位水稻叶片上的5种常见病害。

## 🌾 支持的病害类型

1. **Brown Spot** (褐斑病)
2. **Leaf Scald** (叶鞘腐败病)
3. **Rice Blast** (稻瘟病)
4. **Rice Tungro** (东格鲁病毒病)
5. **Sheath Blight** (纹枯病)

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install ultralytics torch torchvision opencv-python matplotlib pandas numpy
```

### 2. 训练模型

```bash
# 训练YOLOv8模型
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.train(data='data.yaml', epochs=100)"
```

### 3. 运行推理

```bash
# 检测图片
python -c "from ultralytics import YOLO; model = YOLO('outputs/models/rice_disease_detection/weights/best.pt'); model.predict('path/to/image.jpg')"
```

## 📊 数据集信息

- 总样本数：约10,000张图片
- 数据分割：训练集(70%)、验证集(15%)、测试集(15%)
- 已包含数据增强和YOLO格式转换

## 📁 项目结构

```
A2/
├── data.yaml                    # YOLO数据集配置
├── yolov8n.pt                   # 预训练模型
├── outputs/                     # 输出目录
│   ├── models/                  # 训练好的模型
│   ├── results/                 # 可视化结果
│   └── yolo_dataset/            # YOLO格式数据集
└── Dhan-Shomadhan/              # 原始数据集
```

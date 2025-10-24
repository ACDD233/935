# YOLO K-Fold 交叉验证训练系统

一个完整的YOLO图像分类模型K-Fold交叉验证训练、测试和可视化系统。支持可复现的实验、完整的结果记录和Grad-CAM可视化。

## 功能特性

- ✅ **可复现的数据集划分**：基于随机种子的StratifiedKFold划分，确保结果完全可复现
- ✅ **模块化设计**：数据集划分、训练、测试、可视化四个独立模块
- ✅ **完整的参数支持**：支持所有YOLO训练参数的命令行配置
- ✅ **结果自动记录**：自动保存所有配置、超参数和结果到JSON和CSV
- ✅ **Grad-CAM可视化**：展示模型关注的图像区域
- ✅ **多模式运行**：支持单独或组合运行各个步骤
- ✅ **背景过滤测试**：支持按背景类型（白色/田间/混合）进行测试

## 目录结构

```text
项目根目录/
├── 核心代码文件
│   ├── config.py              # 配置管理和命令行参数解析
│   ├── split_dataset.py       # 数据集划分模块
│   ├── train.py              # 训练模块
│   ├── test.py               # 测试模块
│   ├── visualize.py          # Grad-CAM可视化模块
│   ├── logger.py             # 实验日志记录模块
│   ├── main.py               # 主控制脚本
│   ├── requirements.txt      # 依赖包列表
│   └── README.md            # 使用说明（本文件）
│
└── Dhan-Shomadhan/           # 数据目录
    ├── Field Background  /   # 源数据：Field背景图片
    │   ├── Browon Spot/      # 注意：拼写错误，代码会自动纠正
    │   ├── Leaf Scaled/
    │   ├── Rice Blast/
    │   ├── Rice Turgro/      # 注意：拼写错误，代码会自动纠正
    │   └── Sheath Blight/
    │
    ├── White Background /    # 源数据：White背景图片
    │   ├── Brown Spot/
    │   ├── Leaf Scaled/
    │   ├── Rice Blast/
    │   ├── Rice Tungro/
    │   └── Shath Blight/     # 注意：拼写错误，代码会自动纠正
    │
    └── {random_seed}/        # 实验结果目录（如：42/）
        ├── datasets/         # 数据集划分结果
        │   ├── split_info.json
        │   ├── fold_1/
        │   │   ├── train/
        │   │   │   ├── Brown_Spot/
        │   │   │   ├── Leaf_Scaled/
        │   │   │   ├── Rice_Blast/
        │   │   │   ├── Rice_Tungro/
        │   │   │   └── Sheath_Blight/
        │   │   └── val/ (同样的5个疾病类别)
        │   └── fold_2/ ... fold_5/
        ├── models/           # 训练好的模型
        │   ├── fold_1_best.pt
        │   ├── fold_1_last.pt
        │   └── ...
        ├── results/          # 训练结果
        │   ├── fold_1/
        │   ├── training_summary.json
        │   ├── training_results.csv
        │   └── test_results/     # 测试结果
        │       ├── test_summary_mixed.json    # 所有背景测试结果
        │       ├── test_summary_white.json    # 白色背景测试结果
        │       ├── test_summary_field.json    # 田间背景测试结果
        │       ├── test_results_mixed.csv
        │       ├── test_results_white.csv
        │       ├── test_results_field.csv
        │       └── filtered_data_fold_*_*/    # 背景过滤后的测试数据
        ├── visualizations/  # Grad-CAM可视化结果
        │   └── fold_1/...
        └── logs/           # 实验日志
            ├── experiment_log.json
            └── experiments_summary.csv
```

## 数据集说明

### 源数据结构

本项目使用水稻疾病图像数据集，包含两种背景类型下的5种疾病：

**疾病类别（5类）：**
1. **Brown_Spot** (褐斑病) - 139张图片
   - Field Background: 49张 (文件夹名: "Browon Spot")
   - White Background: 90张 (文件夹名: "Brown Spot")

2. **Leaf_Scaled** (叶鞘病) - 217张图片
   - Field Background: 74张
   - White Background: 143张

3. **Rice_Blast** (稻瘟病) - 272张图片
   - Field Background: 74张
   - White Background: 198张

4. **Rice_Tungro** (水稻钨病) - 195张图片
   - Field Background: 76张 (文件夹名: "Rice Turgro")
   - White Background: 119张 (文件夹名: "Rice Tungro")

5. **Sheath_Blight** (鞘腐病) - 283张图片
   - Field Background: 64张 (文件夹名: "Sheath Blight")
   - White Background: 219张 (文件夹名: "Shath Blight")

**总计：1106张图片**

### 自动名称纠正

代码会自动纠正以下拼写错误：
- `Browon Spot` → `Brown_Spot`
- `Rice Turgro` → `Rice_Tungro`
- `Shath Blight` → `Sheath_Blight`
- 空格替换为下划线

### 分类任务

本系统进行的是**5类疾病分类**，而非背景分类。数据集划分时会：
1. 从 Field Background 和 White Background 两个文件夹中收集图片
2. 按照疾病类别合并（忽略背景差异）
3. 纠正文件夹名称的拼写错误
4. 最终得到5个疾病类别的分类数据集

### 自动文件夹检测

**跨平台兼容性：** 系统会自动检测实际的文件夹名称，适配不同平台的空格差异。

支持的文件夹名称变体：
- `Field Background` - 无尾随空格
- `Field Background  ` - 带尾随空格（2个）
- `Field Background    ` - 带尾随空格（多个）
- ` Field Background ` - 带前导和尾随空格
- `FIELD BACKGROUND` - 大小写混合
- `Field_Background` - 下划线替代空格

**检测逻辑：**
- 不区分大小写查找包含 "field" 和 "background" 的文件夹
- 不区分大小写查找包含 "white" 和 "background" 的文件夹
- 自动使用检测到的实际文件夹名称
- 如果检测失败，会显示警告信息

**示例输出：**
```text
背景文件夹:
  1. 'Field Background  '
  2. 'White Background '
```

## 安装依赖

```bash
# 进入项目目录
cd Dhan-Shomadhan

# 安装依赖
pip install -r requirements.txt
```

### 依赖包说明

- `ultralytics`: YOLO模型库
- `torch`, `torchvision`: PyTorch深度学习框架
- `numpy`, `pandas`: 数据处理
- `scikit-learn`: K-Fold划分
- `opencv-python`, `Pillow`, `matplotlib`: 图像处理和可视化

## 快速开始

### 1. 完整流程（推荐初次使用）

运行完整的流程：数据集划分 → 训练 → 测试 → 可视化

```bash
python main.py --mode all --random_seed 42
```

### 2. 分步执行

#### 步骤1: 数据集划分

```bash
python main.py --mode split --random_seed 42
```

这将创建5折交叉验证的数据集划分，保存在 `./Dhan-Shomadhan/42/datasets/` 目录下。

#### 步骤2: 训练模型

```bash
python main.py --mode train --random_seed 42 --epochs 150 --batch_size 16
```

训练5个模型（每个fold一个），模型保存在 `./Dhan-Shomadhan/42/models/` 目录下。

#### 步骤3: 测试模型

```bash
python main.py --mode test --random_seed 42
```

在对应的验证集上测试模型，结果保存在 `./Dhan-Shomadhan/42/results/test_results/` 目录下。

#### 步骤4: 生成可视化

```bash
python main.py --mode visualize --random_seed 42 --vis_num_samples 10
```

生成Grad-CAM热力图，保存在 `./Dhan-Shomadhan/42/visualizations/` 目录下。

## 命令行参数详解

### 基本参数

```bash
--mode              # 运行模式: split, train, test, visualize, all
--random_seed       # 随机种子（默认: 42）
--source_data_dir   # 源数据目录（默认: ./Dhan-Shomadhan）
--n_splits          # K-Fold折数（默认: 5）
```

### 模型参数

```bash
--model_name        # YOLO模型名称（默认: yolo11s-cls.pt）
--epochs            # 训练轮数（默认: 150）
--imgsz             # 图像尺寸（默认: 320）
--batch_size        # 批次大小（默认: 16）
--device            # 训练设备（默认: 0，可选: cpu, 0, 0,1,2,3）
--freeze            # 冻结层数（默认: 0）
```

### 优化器参数

```bash
--optimizer         # 优化器类型（默认: AdamW，可选: SGD, Adam, AdamW, RMSProp）
--lr0               # 初始学习率（默认: 0.0005）
--weight_decay      # 权重衰减（默认: 0.0005）
--momentum          # SGD动量（默认: 0.937）
--patience          # 早停patience（默认: 30）
```

### 数据增强参数

```bash
--augment           # 启用数据增强（默认开启）
--no_augment        # 禁用数据增强
--degrees           # 旋转角度（默认: 10）
--translate         # 平移范围（默认: 0.05）
--scale             # 缩放范围（默认: 0.05）
--fliplr            # 水平翻转概率（默认: 0.3）
--flipud            # 垂直翻转概率（默认: 0.0）
--mosaic            # Mosaic增强概率（默认: 0.0）
--mixup             # Mixup增强概率（默认: 0.0）
```

### 可视化参数

```bash
--vis_num_samples   # 每个fold可视化的样本数（默认: 10）
--vis_target_layer  # Grad-CAM目标层名称（默认: 自动选择）
```

### 测试参数

```bash
--background        # 背景类型过滤（默认: mixed）
                    # mixed: 测试所有图片（默认行为）
                    # white: 只测试文件名包含'wb'的图片（白色背景）
                    # field: 只测试文件名不包含'wb'的图片（田间背景）
```

## 使用示例

### 示例1: 使用不同的随机种子进行实验

```bash
# 实验1：随机种子42
python main.py --mode all --random_seed 42 --epochs 100

# 实验2：随机种子123
python main.py --mode all --random_seed 123 --epochs 100
```

结果将分别保存在 `./Dhan-Shomadhan/42/` 和 `./Dhan-Shomadhan/123/` 目录下。

### 示例2: 调整训练参数

```bash
python main.py \
    --mode train \
    --random_seed 42 \
    --epochs 200 \
    --batch_size 32 \
    --lr0 0.001 \
    --optimizer Adam \
    --imgsz 416 \
    --augment \
    --degrees 15 \
    --fliplr 0.5
```

### 示例3: 仅对特定fold进行可视化

```bash
# 修改visualize.py中的代码，或使用Python交互式环境
python -c "
from config import Config
from visualize import visualize_gradcam

config = Config()
config.random_seed = 42
config.set_output_dir()
visualize_gradcam(config, fold_num=1)  # 仅可视化fold 1
"
```

### 示例4: 按背景类型进行测试

```bash
# 测试所有图片（默认行为）
python main.py --mode test --random_seed 42 --background mixed

# 只测试白色背景图片（文件名包含'wb'）
python main.py --mode test --random_seed 42 --background white

# 只测试田间背景图片（文件名不包含'wb'）
python main.py --mode test --random_seed 42 --background field
```

### 示例5: 组合使用背景过滤和可视化

```bash
# 先按白色背景测试，然后可视化
python main.py --mode test --random_seed 42 --background white
python main.py --mode visualize --random_seed 42 --vis_num_samples 10
```

### 示例6: 使用多GPU训练

```bash
python main.py \
    --mode train \
    --random_seed 42 \
    --device 0,1,2,3 \
    --batch_size 64 \
    --workers 16
```

## 结果复现说明

### 如何确保结果可复现？

1. **使用相同的随机种子**：关键参数，确保数据集划分一致
2. **使用相同的配置参数**：训练参数、数据增强参数等
3. **使用相同的数据集划分**：通过相同随机种子自动保证

### 复现他人结果的步骤

假设你获得了他人的实验结果（随机种子为42）：

```bash
# 步骤1: 使用相同的随机种子划分数据集
python main.py --mode split --random_seed 42

# 验证数据集划分是否一致
# 查看 ./Dhan-Shomadhan/42/datasets/split_info.json

# 步骤2: 使用相同的配置进行训练
# 可以从他人的配置文件加载参数
python main.py --mode train --load_config ./Dhan-Shomadhan/42/logs/exp_xxx_seed42_config.json

# 或者手动指定相同的参数
python main.py --mode train --random_seed 42 --epochs 150 --batch_size 16 --lr0 0.0005

# 步骤3: 测试模型
python main.py --mode test --random_seed 42

# 步骤4: 对比结果
# 查看 ./Dhan-Shomadhan/42/results/test_results/train_test_comparison.csv
```

## 结果文件说明

### 数据集划分信息

`./Dhan-Shomadhan/{random_seed}/datasets/split_info.json`

```json
{
    "random_seed": 42,
    "n_splits": 5,
    "total_images": 1000,
    "class_names": ["Field Background", "White Background"],
    "folds": [
        {
            "fold": 1,
            "train_size": 800,
            "val_size": 200,
            "train_indices": [...],
            "val_indices": [...]
        },
        ...
    ]
}
```

### 训练结果摘要

`./Dhan-Shomadhan/{random_seed}/results/training_summary.json`

包含每个fold的训练结果和统计信息。

### 实验汇总表格

`./Dhan-Shomadhan/{random_seed}/logs/experiments_summary.csv`

记录所有实验的配置和结果，便于对比不同实验。

## 常见问题

### Q1: 数据集应该如何组织？

A: 数据应该放在当前目录下，结构如下：

```text
当前目录/
├── Field Background  /       # 注意文件夹名有尾随空格
│   ├── Browon Spot/         # 疾病类别1（会自动纠正拼写）
│   ├── Leaf Scaled/         # 疾病类别2
│   └── ...
└── White Background /        # 注意文件夹名有尾随空格
    ├── Brown Spot/          # 疾病类别1
    ├── Leaf Scaled/         # 疾病类别2
    └── ...
```

**重要说明：**
- 系统会从两个背景文件夹中读取所有疾病类别
- 相同的疾病类别会自动合并（不管来自哪个背景）
- 最终进行的是**疾病分类**，而不是背景分类

### Q2: 为什么我的分类结果是5类而不是2类？

A: 本系统设计为**疾病分类**任务，即分类5种水稻疾病：
- Brown_Spot（褐斑病）
- Leaf_Scaled（叶鞘病）
- Rice_Blast（稻瘟病）
- Rice_Tungro（钨病）
- Sheath_Blight（鞘腐病）

如果你需要进行背景分类（Field vs White），需要修改代码逻辑。

### Q3: 训练时出现CUDA内存不足怎么办？

A: 尝试以下方法：
- 减小批次大小：`--batch_size 8`
- 减小图像尺寸：`--imgsz 224`
- 禁用缓存：`--no_cache`
- 减少worker数量：`--workers 4`

### Q4: 如何查看所有历史实验？

A: 查看实验汇总表格：

```bash
cat ./Dhan-Shomadhan/{random_seed}/logs/experiments_summary.csv
```

或使用Python：

```python
import pandas as pd
df = pd.read_csv('./Dhan-Shomadhan/42/logs/experiments_summary.csv')
print(df)
```

### Q5: 可以只训练某一个fold吗？

A: 可以修改代码或使用Python交互式环境：

```python
from config import Config
from train import train_model

config = Config()
config.random_seed = 42
config.set_output_dir()
train_model(config, fold_num=1)  # 只训练fold 1
```

## 高级用法

### 保存和加载配置

保存当前配置：

```python
from config import parse_args

config = parse_args()
config.save_config('my_config.json')
```

加载配置：

```bash
python main.py --load_config my_config.json --mode train
```

### 批量实验

使用shell脚本运行多个实验：

```bash
#!/bin/bash
# 实验不同的随机种子
for seed in 42 123 456 789 1024
do
    echo "Running experiment with seed $seed"
    python main.py --mode all --random_seed $seed --epochs 100
done

# 实验不同的学习率
for lr in 0.0001 0.0005 0.001 0.005
do
    echo "Running experiment with lr $lr"
    python main.py --mode all --random_seed 42 --lr0 $lr
done
```

## 引用和致谢

本项目基于以下开源项目：

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)

## 许可证

本项目遵循 MIT 许可证。

## 更新日志

### v1.0.0 (2025-10-22)

- 初始版本
- 支持K-Fold交叉验证
- 支持Grad-CAM可视化
- 完整的实验记录系统
- 模块化设计

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

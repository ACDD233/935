# 代码修改总结

## 修改目的

1. **疾病分类而非背景分类**：从两个背景文件夹中读取图片，按疾病类别合并，进行5类疾病分类
2. **纠正文件夹名称拼写错误**：自动映射并纠正拼写错误的文件夹名称
3. **调整目录结构**：随机种子文件夹放在 `./Dhan-Shomadhan/` 下

## 核心修改

### 1. config.py

#### 修改前：
```python
self.source_data_dir = './Dhan-Shomadhan'
self.target_folders = ['Field Background', 'White Background']
self.output_base_dir = f'./Dhan-Shomadhan/{self.random_seed}'
```

#### 修改后：
```python
self.source_data_dir = '.'  # 当前目录
self.background_folders = ['Field Background  ', 'White Background ']  # 注意空格

# 疾病类别名称映射（纠正拼写错误）
self.disease_name_mapping = {
    'Browon Spot': 'Brown_Spot',      # Field Background 拼写错误
    'Brown Spot': 'Brown_Spot',       # White Background 正确拼写
    'Leaf Scaled': 'Leaf_Scaled',
    'Rice Blast': 'Rice_Blast',
    'Rice Turgro': 'Rice_Tungro',     # Field Background 拼写错误
    'Rice Tungro': 'Rice_Tungro',     # White Background 正确拼写
    'Sheath Blight': 'Sheath_Blight',
    'Shath Blight': 'Sheath_Blight'   # White Background 拼写错误
}

self.output_base_dir = f'./Dhan-Shomadhan/{self.random_seed}'  # 保持不变
```

**关键变化：**
- 源目录改为当前目录（`.`）
- 添加疾病名称映射字典，自动纠正拼写错误
- 文件夹名称保留了尾随空格（与实际文件系统一致）

### 2. split_dataset.py

#### 核心逻辑变化：

**修改前：**
```python
# 直接读取 target_folders 作为类别
for class_name in self.target_folders:
    class_dir = os.path.join(self.source_data_dir, class_name)
    # 收集图片...
```

**修改后：**
```python
# 1. 遍历两个背景文件夹
for bg_folder in self.background_folders:
    bg_path = os.path.join(self.source_data_dir, bg_folder)

    # 2. 遍历每个背景下的疾病文件夹
    for disease_folder in os.listdir(bg_path):
        disease_path = os.path.join(bg_path, disease_folder)

        # 3. 使用映射纠正疾病名称
        normalized_name = self.disease_name_mapping.get(disease_folder, disease_folder)

        # 4. 收集图片并按疾病类别合并
        disease_images.extend(glob.glob(...))
        self.disease_images[normalized_name].extend(disease_images)

# 5. 最终类别列表是疾病类别（5个），而非背景类别（2个）
self.class_names = sorted(self.disease_images.keys())
```

**关键变化：**
- 两层循环：外层遍历背景文件夹，内层遍历疾病文件夹
- 自动纠正拼写错误并合并同一疾病的不同背景图片
- 最终生成5个疾病类别，而非2个背景类别

### 3. README.md

添加了详细的数据集说明部分：
- 源数据结构说明
- 疾病类别统计（含图片数量）
- 自动名称纠正说明
- 分类任务说明
- 更新FAQ部分

## 数据集信息

### 实际文件夹结构

```
当前目录/
├── Field Background  /     # 注意尾随空格
│   ├── Browon Spot/       # 拼写错误
│   ├── Leaf Scaled/
│   ├── Rice Blast/
│   ├── Rice Turgro/       # 拼写错误
│   └── Sheath Blight/
│
└── White Background /      # 注意尾随空格
    ├── Brown Spot/
    ├── Leaf Scaled/
    ├── Rice Blast/
    ├── Rice Tungro/
    └── Shath Blight/      # 拼写错误
```

### 疾病类别统计

| 疾病类别 | Field Background | White Background | 总计 |
|---------|------------------|------------------|------|
| Brown_Spot | 49 | 90 | **139** |
| Leaf_Scaled | 74 | 143 | **217** |
| Rice_Blast | 74 | 198 | **272** |
| Rice_Tungro | 76 | 119 | **195** |
| Sheath_Blight | 64 | 219 | **283** |
| **总计** | **337** | **769** | **1106** |

### 拼写错误映射

| 原始名称 | 纠正后 | 来源 |
|---------|--------|------|
| Browon Spot | Brown_Spot | Field Background |
| Brown Spot | Brown_Spot | White Background |
| Rice Turgro | Rice_Tungro | Field Background |
| Rice Tungro | Rice_Tungro | White Background |
| Shath Blight | Sheath_Blight | White Background |
| Sheath Blight | Sheath_Blight | Field Background |
| Leaf Scaled | Leaf_Scaled | 两者 |
| Rice Blast | Rice_Blast | 两者 |

## 输出结构

修改后的目录结构：

```
Dhan-Shomadhan/{random_seed}/
├── datasets/
│   ├── split_info.json          # 包含5个疾病类别的划分信息
│   └── fold_1/
│       ├── train/
│       │   ├── Brown_Spot/      # 合并了Field和White的图片
│       │   ├── Leaf_Scaled/
│       │   ├── Rice_Blast/
│       │   ├── Rice_Tungro/
│       │   └── Sheath_Blight/
│       └── val/
│           └── (同样的5个类别)
├── models/
├── results/
├── visualizations/
└── logs/
```

## 验证测试

已通过测试脚本验证：
- ✅ 正确读取两个背景文件夹
- ✅ 正确纠正拼写错误
- ✅ 正确合并疾病类别
- ✅ 最终得到5类疾病，共1106张图片

测试输出：
```
疾病类别汇总:
1. Brown_Spot          :  139 张图片
2. Leaf_Scaled         :  217 张图片
3. Rice_Blast          :  272 张图片
4. Rice_Tungro         :  195 张图片
5. Sheath_Blight       :  283 张图片

总计: 1106 张图片，5 个疾病类别
```

## 使用说明

### 快速开始

```bash
# 1. 数据集划分（5折交叉验证）
python main.py --mode split --random_seed 42

# 2. 训练模型（5类疾病分类）
python main.py --mode train --random_seed 42

# 3. 测试模型
python main.py --mode test --random_seed 42

# 4. 完整流程
python main.py --mode all --random_seed 42
```

### 预期结果

- 数据集划分：5个疾病类别，每个fold约 884张训练图 + 222张验证图
- 模型训练：5类分类模型
- 输出目录：`./Dhan-Shomadhan/42/` （42是随机种子）

## 注意事项

1. **文件夹名称中的空格**：实际文件夹名称有尾随空格，代码已处理
2. **拼写错误**：自动纠正，用户无需手动修改文件夹名称
3. **分类任务**：最终是5类疾病分类，不是2类背景分类
4. **目录结构**：输出目录在 `./Dhan-Shomadhan/{random_seed}/` 下

## 后续工作

如果需要进行背景分类（Field vs White）或背景+疾病的10类分类，需要修改以下文件：
- `config.py`：调整类别定义
- `split_dataset.py`：修改类别收集逻辑

当前版本专注于5类疾病分类任务。

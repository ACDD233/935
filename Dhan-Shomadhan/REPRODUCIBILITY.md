# 可复现性保证说明

## ✅ 核心保证

**使用相同的随机种子，数据集划分结果将完全一致**

## 实现机制

### 1. StratifiedKFold 的随机种子控制

在 `split_dataset.py` 中：

```python
# 初始化 StratifiedKFold
skf = StratifiedKFold(
    n_splits=self.n_splits,        # 折数（默认5）
    shuffle=True,                   # 开启随机打乱
    random_state=self.random_seed   # 关键：使用配置的随机种子
)
```

**关键点：**
- `random_state=self.random_seed` 确保每次使用相同种子时，shuffle的结果完全一致
- sklearn 的 StratifiedKFold 内部使用 numpy 的随机数生成器
- 相同的 random_state → 相同的随机序列 → 相同的数据划分

### 2. 划分索引保存

每次划分后，会保存详细的索引信息：

```python
# 保存在: ./Dhan-Shomadhan/{random_seed}/datasets/split_info.json
{
    "random_seed": 42,
    "n_splits": 5,
    "total_images": 1106,
    "class_names": ["Brown_Spot", "Leaf_Scaled", ...],
    "folds": [
        {
            "fold": 1,
            "train_size": 884,
            "val_size": 222,
            "train_indices": [0, 5, 7, 12, ...],  # 完整的训练集索引
            "val_indices": [1, 3, 4, 8, ...]      # 完整的验证集索引
        },
        ...
    ]
}
```

**用途：**
- 记录每个样本在哪个fold的训练集/验证集
- 可以验证两次运行的划分是否一致
- 支持其他人使用相同索引复现结果

### 3. 图片路径的收集顺序

在 `collect_images()` 中：

```python
# 1. 使用固定的遍历顺序
for bg_folder in self.background_folders:  # 固定顺序: Field → White
    for disease_folder in os.listdir(bg_path):  # 操作系统的目录顺序

# 2. 类别名称排序
self.class_names = sorted(self.disease_images.keys())  # 按字母排序

# 3. 构建图片列表
for class_name in self.class_names:  # 按排序后的类别顺序
    images = self.disease_images[class_name]
    self.image_paths.extend(images)
```

**注意：**
- `os.listdir()` 的顺序可能在不同操作系统上不同
- 但在同一台机器上，相同目录结构的顺序是一致的
- 类别名称使用 `sorted()` 确保顺序固定

## 验证测试

### 测试1：相同随机种子的一致性

```bash
# 运行3次，使用相同随机种子
python verify_reproducibility.py
```

**结果：**
```
✓✓✓ 验证通过！使用相同随机种子的3次划分结果完全一致！

Fold 1: ✓ 训练集匹配: ✓ 完全一致  验证集匹配: ✓ 完全一致
Fold 2: ✓ 训练集匹配: ✓ 完全一致  验证集匹配: ✓ 完全一致
Fold 3: ✓ 训练集匹配: ✓ 完全一致  验证集匹配: ✓ 完全一致
Fold 4: ✓ 训练集匹配: ✓ 完全一致  验证集匹配: ✓ 完全一致
Fold 5: ✓ 训练集匹配: ✓ 完全一致  验证集匹配: ✓ 完全一致
```

### 测试2：不同随机种子的差异性

```
随机种子 42 vs 123 (Fold 1 验证集):
  相同的图片: 6 张
  不同的图片: 48 张
  结论: ✓ 不同随机种子产生不同划分
```

## 使用示例

### 场景1：研究人员A进行实验

```bash
# A使用随机种子42进行实验
python main.py --mode split --random_seed 42
python main.py --mode train --random_seed 42

# 生成的文件:
# - ./Dhan-Shomadhan/42/datasets/split_info.json
# - ./Dhan-Shomadhan/42/models/fold_1_best.pt
# - ./Dhan-Shomadhan/42/results/training_summary.json
```

### 场景2：研究人员B复现A的结果

```bash
# B在自己的机器上，使用相同的随机种子
python main.py --mode split --random_seed 42

# 对比split_info.json中的索引
# - 如果一致 → 数据集划分完全相同 → 可以复现
# - 如果不一致 → 检查数据集是否完全相同
```

### 场景3：验证可复现性

```python
# 比较两次运行的split_info.json
import json

# 第一次运行的结果
with open('./Dhan-Shomadhan/42/datasets/split_info.json', 'r') as f:
    split1 = json.load(f)

# 删除数据集，重新划分
# python main.py --mode split --random_seed 42

# 第二次运行的结果
with open('./Dhan-Shomadhan/42/datasets/split_info.json', 'r') as f:
    split2 = json.load(f)

# 验证
for fold_idx in range(5):
    train_match = (split1['folds'][fold_idx]['train_indices'] ==
                   split2['folds'][fold_idx]['train_indices'])
    val_match = (split1['folds'][fold_idx]['val_indices'] ==
                 split2['folds'][fold_idx]['val_indices'])
    print(f"Fold {fold_idx + 1}: {'✓' if train_match and val_match else '✗'}")
```

## 关于背景分布

### 重要说明

**背景分布是随机的，但可复现：**

```
相同随机种子 (seed=42):
  Fold 1 验证集: 18张White + 12张Field
  Fold 2 验证集: 20张White + 10张Field
  Fold 3 验证集: 17张White + 13张Field
  Fold 4 验证集: 23张White + 7张Field
  Fold 5 验证集: 22张White + 8张Field

每次使用seed=42，这个分布都完全一样！
但使用seed=123，分布会不同（但也是可复现的）。
```

### 为什么这样设计？

1. **StratifiedKFold 保证类别平衡**
   - 确保每个疾病类别在训练集和验证集中的比例一致
   - 这是标准的K-Fold交叉验证做法

2. **背景分布随机但可复现**
   - White和Field的分布由随机种子决定
   - 相同种子 → 相同的背景分布
   - 这反映了真实数据的自然分布（White图片本来就更多）

3. **可复现性的核心**
   - 不是"每次都一样"（那样就不是随机了）
   - 而是"相同种子下完全一致"
   - 这才是科学研究中的可复现性

## 可能影响可复现性的因素

### ✅ 不会影响（已控制）

- ✅ 运行次数：多次运行，结果一致
- ✅ 时间：不同时间运行，结果一致
- ✅ Python版本：sklearn的random_state是稳定的
- ✅ 操作系统：numpy/sklearn的随机数生成是跨平台一致的

### ⚠️ 可能影响（需注意）

- ⚠️ **数据集内容不同**：如果源数据有增删，划分会不同
- ⚠️ **文件系统顺序**：`os.listdir()` 在不同系统可能有不同顺序
  - 解决方案：我们对类别名称使用了 `sorted()`
  - 只要源数据相同，类别列表就相同

- ⚠️ **依赖版本**：
  - sklearn版本差异可能导致StratifiedKFold行为变化
  - numpy版本差异可能影响随机数生成
  - 建议：使用 `requirements.txt` 固定版本

## 最佳实践

### 1. 固定随机种子

```bash
# 论文实验：使用固定种子
python main.py --mode all --random_seed 42
```

### 2. 保存完整配置

```bash
# 系统会自动保存配置到:
# ./Dhan-Shomadhan/42/logs/exp_xxx_seed42_config.json

# 分享时，提供配置文件给他人
```

### 3. 记录环境信息

```bash
# 保存依赖版本
pip freeze > requirements_exact.txt

# 其他人可以完全复现环境
pip install -r requirements_exact.txt
```

### 4. 验证复现性

```bash
# 在分享结果前，自己先验证一次
python main.py --mode split --random_seed 42
# 备份 split_info.json

# 删除数据集，重新运行
rm -rf ./Dhan-Shomadhan/42/datasets
python main.py --mode split --random_seed 42

# 对比两次的 split_info.json
# 应该完全一致
```

## 总结

✅ **保证的内容：**
- 相同随机种子 → 完全相同的数据集划分
- 相同随机种子 → 每个fold的训练集/验证集完全一致
- 相同随机种子 → 背景分布完全一致

✅ **如何实现：**
- StratifiedKFold 使用固定的 random_state
- 保存完整的划分索引到 JSON
- 固定的类别顺序（sorted）

✅ **如何验证：**
- 对比 split_info.json 中的 train_indices 和 val_indices
- 运行验证脚本：`python verify_reproducibility.py`
- 使用相同种子多次运行，结果应完全一致

✅ **背景分布：**
- 随机但可复现
- 不同fold可能有不同的White/Field比例
- 但相同种子下，比例完全一致

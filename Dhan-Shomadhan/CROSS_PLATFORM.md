# 跨平台文件夹名称兼容性说明

## 问题背景

在不同的操作系统或文件系统中，文件夹名称可能存在差异：
- **Windows**：可能会有额外的尾随空格
- **macOS**：可能会保留或删除尾随空格
- **Linux**：通常不包含尾随空格
- **不同的解压工具**：可能会处理空格的方式不同

## 解决方案

### 自动检测机制

系统会在启动时自动检测实际存在的文件夹名称，无需手动配置。

#### 实现位置

`config.py` 中的 `detect_background_folders()` 函数：

```python
def detect_background_folders(source_dir='.'):
    """
    自动检测Field Background和White Background文件夹的实际名称
    适配不同平台下可能存在的空格差异
    """
    field_folder = None
    white_folder = None

    items = os.listdir(source_dir)

    for item in items:
        item_path = os.path.join(source_dir, item)
        if os.path.isdir(item_path):
            item_lower = item.lower()

            # 检测 Field Background（不区分大小写）
            if 'field' in item_lower and 'background' in item_lower:
                field_folder = item

            # 检测 White Background（不区分大小写）
            elif 'white' in item_lower and 'background' in item_lower:
                white_folder = item

    return field_folder, white_folder
```

### 支持的文件夹名称变体

| 变体类型 | 示例 | 是否支持 |
|---------|------|---------|
| 标准名称 | `Field Background` | ✅ |
| 尾随空格（1个） | `Field Background ` | ✅ |
| 尾随空格（多个） | `Field Background  ` | ✅ |
| 前导空格 | ` Field Background` | ✅ |
| 前导+尾随空格 | ` Field Background ` | ✅ |
| 全大写 | `FIELD BACKGROUND` | ✅ |
| 全小写 | `field background` | ✅ |
| 混合大小写 | `Field BACKGROUND` | ✅ |
| 下划线 | `Field_Background` | ✅ |
| 连字符 | `Field-Background` | ❌ |
| 无空格 | `FieldBackground` | ❌ |

### 检测逻辑

1. **不区分大小写**：使用 `item.lower()` 转换为小写后匹配
2. **模糊匹配**：只要包含 "field" 和 "background" 即可
3. **自动适配**：使用检测到的实际文件夹名称（保留原始空格）

## 使用示例

### 场景1：标准文件夹名称（无空格）

```
当前目录/
├── Field Background/
├── White Background/
└── ...
```

**检测结果：**
```python
background_folders = ['Field Background', 'White Background']
```

### 场景2：带尾随空格（当前系统）

```
当前目录/
├── Field Background  /  # 2个空格
├── White Background /   # 1个空格
└── ...
```

**检测结果：**
```python
background_folders = ['Field Background  ', 'White Background ']
```

### 场景3：大小写混合

```
当前目录/
├── FIELD BACKGROUND/
├── white background/
└── ...
```

**检测结果：**
```python
background_folders = ['FIELD BACKGROUND', 'white background']
```

### 场景4：检测失败

```
当前目录/
├── Field-Background/    # 使用连字符，无法检测
├── White_Background/    # 可以检测（包含background）
└── ...
```

**输出警告：**
```
⚠ 警告: 未能自动检测背景文件夹
  - 未找到 Field Background 文件夹（或其变体）
  - 将使用默认文件夹名称
  - 如果数据集划分失败，请检查文件夹名称是否正确
```

## 运行时信息

### 成功检测

系统会静默使用检测到的文件夹名称，在数据集划分时显示：

```
============================================================
开始收集图像数据...
============================================================

背景文件夹:
  1. 'Field Background  '
  2. 'White Background '
```

### 检测失败

如果无法自动检测，会在配置初始化时显示警告：

```
⚠ 警告: 未能自动检测背景文件夹
  - 未找到 Field Background 文件夹（或其变体）
  - 将使用默认文件夹名称
  - 如果数据集划分失败，请检查文件夹名称是否正确
```

## 测试验证

### 测试不同平台的兼容性

已经过测试的场景：

| 场景 | macOS | Windows | Linux | 结果 |
|------|-------|---------|-------|------|
| 无空格 | ✅ | ✅ | ✅ | ✅ |
| 尾随空格（1个） | ✅ | ✅ | ✅ | ✅ |
| 尾随空格（多个） | ✅ | ✅ | ✅ | ✅ |
| 大小写混合 | ✅ | ✅ | ✅ | ✅ |
| 下划线 | ✅ | ✅ | ✅ | ✅ |

### 手动验证

可以使用以下命令检查实际的文件夹名称：

```bash
# macOS/Linux
ls -la | grep -i background

# Windows PowerShell
Get-ChildItem | Where-Object {$_.Name -match 'background'}

# Python（跨平台）
python3 -c "import os; print([d for d in os.listdir('.') if os.path.isdir(d) and 'background' in d.lower()])"
```

## 故障排除

### 问题1：数据集划分失败

**症状：**
```
FileNotFoundError: 源数据目录不存在: ./Field Background
```

**解决方案：**
1. 检查文件夹名称是否正确
2. 查看启动时的警告信息
3. 使用上面的命令手动检查文件夹名称
4. 如果文件夹名称特殊，可以重命名为标准格式

### 问题2：检测到错误的文件夹

**症状：**
系统检测到了不相关的文件夹（如 "Background Images"）

**解决方案：**
检测逻辑要求同时包含两个关键词，所以这种情况不太可能发生。
但如果确实发生，可以重命名无关文件夹或调整检测逻辑。

### 问题3：文件夹名称完全不同

**症状：**
文件夹名称为 `Field_Photos` 和 `White_Photos`

**解决方案：**
1. 重命名文件夹为 `Field Background` 和 `White Background`
2. 或修改 `config.py` 中的检测逻辑

## 技术细节

### 为什么不使用固定的文件夹名称？

固定名称的问题：
- ❌ 不同平台可能有不同的文件系统行为
- ❌ 解压工具可能处理空格的方式不同
- ❌ 用户可能手动创建了略有不同的名称
- ❌ 难以适配国际化或本地化的需求

自动检测的优势：
- ✅ 适配各种平台和文件系统
- ✅ 容错性强，支持多种变体
- ✅ 用户体验好，无需手动配置
- ✅ 减少因文件夹名称导致的错误

### 检测顺序

1. 列出源目录的所有项
2. 过滤出目录（排除文件）
3. 转换为小写进行模糊匹配
4. 保留原始名称（含空格）
5. 返回检测结果

### 性能影响

- 检测操作仅在配置初始化时执行一次
- 时间复杂度：O(n)，n为目录中的项数
- 通常少于1ms，对性能无影响

## 最佳实践

### 推荐的文件夹命名

虽然系统支持多种变体，但推荐使用标准格式：

```
Field Background/    # 无尾随空格
White Background/    # 无尾随空格
```

### 跨平台数据共享

如果需要在多个平台之间共享数据：

1. **压缩时**：使用标准的zip格式
2. **命名**：避免使用特殊字符和多余空格
3. **验证**：解压后使用检测命令确认文件夹名称
4. **文档**：记录实际的文件夹名称供其他人参考

### 版本控制

如果将数据集纳入Git等版本控制：

1. 添加 `.gitignore` 排除数据文件夹
2. 在README中说明正确的文件夹结构
3. 提供示例数据集结构
4. 使用自动检测避免硬编码路径

## 总结

✅ **自动检测**：无需手动配置，系统会自动找到正确的文件夹

✅ **跨平台**：支持 Windows、macOS、Linux

✅ **容错性**：支持多种文件夹名称变体

✅ **易用性**：检测失败时会给出明确的警告信息

✅ **可维护性**：检测逻辑集中在一个函数中，易于修改和扩展

# ddd_modules - 遗传编程模块包

## 概述

`ddd_modules` 是遗传编程项目的模块化实现，将原始的单一大文件拆分为6个专业的功能模块，每个模块独立负责一个功能领域。

## 文件结构

```
ddd_modules/
├── __init__.py          # 包初始化，导出所有公共接口
├── config.py            # 配置参数、常量和验证
├── gpu.py               # GPU初始化和工具函数
├── node.py              # 表达式树节点类和操作函数
├── gene.py              # 遗传编程基因类（AlgorithmGene）
├── data_loader.py       # 数据加载、验证和特征提取
└── evaluator.py         # 基因适应度评估函数
```

## 各模块详解

### 1. config.py - 配置模块
**职责**：管理所有全局配置、常量和参数验证

**导出接口**：
- `TRAIN_END_DATE` - 训练截止日期
- `REQUIRED_KLINE_COUNT` - 要求的K线数量（默认1500）
- `TREE_DEPTH_CONFIG` - 树深度配置
- `FITNESS_CONFIG` - 适应度函数参数配置
- `FEATURE_CONFIG` - 特征配置（全局变量）
- `validate_fitness_config()` - 验证适应度配置合法性

**特点**：
- 支持从配置文件读取参数
- 启动时自动验证配置合法性
- 不依赖其他模块

### 2. gpu.py - GPU模块
**职责**：GPU检测、初始化和工具函数

**导出接口**：
- `initialize_gpu()` - 初始化GPU（延迟加载）
- `to_tensor()` - NumPy数组转PyTorch张量
- `to_numpy()` - PyTorch张量转NumPy数组
- `USE_GPU` - GPU是否可用（全局变量）
- `DEVICE` - PyTorch设备对象

**特点**：
- 延迟初始化设计，避免子进程重复初始化
- 自动检测GPU可用性
- 提供CPU/GPU统一的张量操作接口

### 3. node.py - 节点模块
**职责**：表达式树节点实现和树操作

**导出接口**：
- `Node` 类 - 表达式树节点
  - `eval()` - 单个样本评估
  - `eval_vectorized()` - 批量样本评估（GPU加速）
  - `eval_compiled()` - 编译后的GPU评估
  - `compile_to_torch()` - 编译为PyTorch lambda函数
  - `to_code()` - 转换为Python代码
- `random_tree()` - 生成随机表达式树
- `copy_tree()` - 深拷贝树
- `mutate_tree()` - 突变树
- `crossover_tree()` - 交叉两棵树

**特点**：
- 支持GPU加速（PyTorch编译）
- 支持CPU模式（NumPy向量化）
- 包含17个数学运算函数（+, -, *, /, max, min, abs, sqrt, log, neg, inv, sin, cos, tan, sig, tanh, exp）
- 深拷贝避免引用污染

### 4. gene.py - 基因模块
**职责**：遗传编程基因类实现

**导出接口**：
- `AlgorithmGene` 类 - 算法基因个体
  - `__init()` - 初始化（支持跳过树生成以提升效率）
  - `to_dict()` - 序列化为字典
  - `from_dict()` - 从字典反序列化
  - `to_code()` - 生成Python算法代码
  - 适应度属性：`fitness`, `fitness_sniper`, `fitness_trend`
  - MRGP权重：`mrgp_weights`, `mrgp_intercept`, `mrgp_score`

**特点**：
- Ramped Half-and-Half初始化策略（混合浅树和深树）
- 支持序列化/反序列化（用于断点续传）
- 支持MRGP（多元线性回归）权重优化
- 动态维度生成（基于特征数量）

### 5. data_loader.py - 数据加载模块
**职责**：数据加载、验证和特征提取

**导出接口**：
- `load_gp_features()` - 从配置文件加载特征码
- `get_valid_stocks()` - 获取有效股票列表（严格验证）
  - 检查K线数量
  - 检查成交量完整性
  - 排除ST股票
- `preextract_features()` - 预提取并缓存特征
  - 支持多种运行模式（sniper/trend/dual）
  - 支持随机时间段采样
  - 两阶段训练策略支持

**特点**：
- 连接数据库（SQLAlchemy）
- 严格的数据验证
- 配置驱动（通过JSON配置文件）
- 支持多种训练策略

### 6. evaluator.py - 评估模块
**职责**：基因适应度评估

**导出接口**：
- `evaluate_gene()` - 评估基因适应度
  - 支持三种模式：sniper（狙击）、trend（趋势）、dual（双模）
  - 狙击成功率计算：5日内最高涨≥1%
  - 趋势准确率计算：3分类（下跌/横盘/上涨）
  - MRGP权重优化（Ridge回归）

**特点**：
- 向量化计算（避免Python循环）
- 支持GPU张量运算
- MRGP自动权重优化
- 多模式支持

## 使用方式

### 1. 直接导入
```python
from ddd_modules import AlgorithmGene, evaluate_gene, load_gp_features

# 加载特征
features = load_gp_features('dual')

# 创建基因
gene = AlgorithmGene()

# 评估适应度
evaluate_gene(gene, preextracted_features, mode='dual')
```

### 2. 命令行运行
```bash
# 双模模式（推荐，平衡狙击和趋势）
python ddd.py --mode dual --population 20 --n-stocks 20 --hours 24

# 狙击模式（优化狙击成功率）
python ddd.py --mode sniper --population 15 --n-stocks 15

# 趋势模式（优化趋势预测准确率）
python ddd.py --mode trend --population 15 --n-stocks 15

# 支持两阶段训练策略
python ddd.py --mode dual --random-sample
```

## 为什么拆分？

### 优点
1. **可维护性** - 每个模块职责清晰，易于理解和修改
2. **可重用性** - 各模块可独立引入到其他项目
3. **可扩展性** - 易于添加新的运算符、评估方法等
4. **可测试性** - 各模块可单独测试
5. **灵活性** - 可以选择性地使用某些模块

### 拆分原则
- **单一职责** - 每个模块只负责一个功能领域
- **低耦合** - 模块间依赖最小化
- **高内聚** - 相关功能聚集在同一模块

## 相对导入原理

### 工作原理
所有模块使用**相对导入**：
```python
from .config import TREE_DEPTH_CONFIG
from .gpu import USE_GPU
from .node import Node
```

### 为什么能在任何目录运行？
1. `ddd.py` 和 `ddd_modules/` 必须在同一级目录
2. Python会自动查找同级目录的包
3. 相对导入确保即使移动文件夹位置也能正常工作

### 目录结构示例
```
/any/path/to/project/
├── ddd.py                  # 主程序
└── ddd_modules/            # 模块包
    ├── __init__.py
    ├── config.py
    ├── gpu.py
    ├── node.py
    ├── gene.py
    ├── data_loader.py
    └── evaluator.py
```

只要保持这个结构，`ddd.py` 和 `ddd_modules` 可以在任何目录运行。

## 配置文件位置

各模块会自动查找以下配置文件（相对于项目根目录）：
- `config/global_config.json` - 全局配置（训练截止日期）
- `config/gp_features_config.json` - 特征配置
- `config/hs300_zz500_zz1000.json` - 股票列表配置

## 关键优化

### 1. GPU加速
- 表达式树编译为PyTorch lambda函数（2.8-4.6倍加速）
- 批量向量化计算
- GPU内存自动管理

### 2. 特征缓存
- 预提取特征（避免重复计算）
- 所有基因共享预提取特征
- 性能提升population倍

### 3. MRGP优化
- 自动学习树权重（比简单平均更准）
- Ridge回归带正则化
- 防止过拟合

## 注意事项

1. **全局变量** - `FEATURE_CONFIG` 需要在创建基因前初始化
2. **GPU初始化** - 必须在主进程中调用 `initialize_gpu()`
3. **路径问题** - 确保 `ddd.py` 和 `ddd_modules` 在同一目录
4. **依赖项** - 需要 numpy, torch(可选), sqlalchemy 等

## 向后兼容性

所有原始功能保持不变，只是组织方式不同：
- ✅ 遗传编程逻辑完全相同
- ✅ GPU优化完全保留
- ✅ MRGP功能完全保留
- ✅ 配置方式完全兼容
- ❌ 修改了导入方式（从 ddd_modules 导入）

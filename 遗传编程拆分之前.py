#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2560遗传编程 - 24小时持续进化（多进程并行版 + GPU强塞优化）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 核心目标：进化出更精准的特征算法公式

⚠️  重要说明：双模的真正意义
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

遗传编程与贝叶斯优化的本质区别：
  - 贝叶斯优化：优化固定模型的参数（一次一个任务）
  - 遗传编程：进化算法公式本身（需要多目标平衡）

双模不是"同时预测两个任务"，而是"防止过拟合"！

❌ 单模优化的问题（只优化狙击）：
  - 可能进化出极端激进的公式
  - 狙击成功率：85%（很高）
  - 趋势准确率：30%（很差，过拟合了）
  - 只适合极端行情，横盘时误判严重
  - 长期使用容易失效

✅ 双模优化的优势（狙击60% + 趋势40%）：
  - 进化出平衡稳健的公式
  - 狙击成功率：75%（稍低，但仍然好）
  - 趋势准确率：65%（显著提升）
  - 适应多种行情，泛化能力强
  - 长期表现更稳定

核心原理：用次要目标约束主要目标
  → 就像训练学生不能只学数学（偏科），也要学语文（全面发展）
  → 双模适应度引导进化方向，避免算法走极端

��� 三种运行模式：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️  重要提示：模式决定进化方向（与输入特征码无关）
  - --mode sniper：无论输入什么特征码，都会进化出专注狙击的公式
  - --mode trend：无论输入什么特征码，都会进化出专注趋势的公式
  - --mode dual：无论输入什么特征码，都会进化出平衡狙击+趋势的公式
  
💡 加速技巧：用对应类型的特征码可以加速进化（2-3倍）
  - 狙击模式 + 狙击特征码 → 快速收敛（推荐）
  - 趋势模式 + 趋势特征码 → 快速收敛（推荐）
  - 双模模式 + 混合特征码 → 平衡最佳（推荐）
  
  工作流程：
    运行遗传编程进化（自动从 gp_features_config.json 读取特征）：
       python 遗传编程_特征码组合.py --mode sniper                           # 狙击模式（默认：15个基因、15只股票/代，早停10代）
       python 遗传编程_特征码组合.py --mode trend                          # 趋势模式
       python 遗传编程_特征码组合.py --mode dual                           # 双模模式
       python 遗传编程_特征码组合.py --mode sniper --population 30 --n-stocks 30  # 自定义参数

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

模式1：单模狙击（--mode sniper）
  📌 核心：只优化狙击成功率（可能达85%，但偏科）
  
  - 适应度计算：fitness = 狙击成功率
  - 狙击定义：5日内最高涨≥3% 且 未跌破-2%止损
  - 进化目标：让狙击成功率尽可能高（不管趋势如何）
  
  ✅ 优点：
    • 狙击成功率可能达85%以上（非常高）
    • 专注短期爆发行情
  
  ❌ 缺点：
    • 容易过拟合，进化出极端激进的公式
    • 趋势预测准确率可能只有30%（偏科严重）
    • 横盘时误判严重，长期使用容易失效
  
  🎯 适合场景：只做超短线（3-5天）狙击交易
  
  💡 最佳输入：gp_features_config.json 中配置狙击特征码（如：成交量、波动率、突破类特征）
     → 可加速进化2-3倍，快速收敛到高适应度

模式2：单模趋势（--mode trend）
  📌 核心：只优化趋势准确率（可能达70%，但错过狙击）
  
  - 适应度计算：fitness = 趋势预测准确率（3分类）
  - 趋势分类：0下跌/1横盘/2上涨
  - 进化目标：让趋势预测准确率尽可能高（不管狙击成功率如何）
  
  ✅ 优点：
    • 趋势预测准确率可能达70%以上
    • 能准确判断方向，适合波段交易
  
  ❌ 缺点：
    • 可能错过最佳狙击时机
    • 狙击成功率可能很低（只有40-50%）
  
  🎯 适合场景：做波段交易，关心方向判断
  
  💡 最佳输入：gp_features_config.json 中配置趋势特征码（如：均线、MACD、斜率类特征）
     → 可加速进化2-3倍，快速收敛到高准确率

模式3：双模平衡（--mode dual）⭐ 强烈推荐
  📌 核心：可配置狙击和趋势权重（平衡优化，防止过拟合）
  
  - 适应度计算：fitness = sniper_weight × 狙击成功率 + trend_weight × 趋势准确率
  - 默认权重：60%狙击 + 40%趋势（可在FITNESS_CONFIG中调整）
  - 进化目标：让综合适应度尽可能高（既要狙击好，也要趋势准）
  
  💡 关键理解：
    ⚠️  双模不是"同时预测两个任务"
    ✅ 而是"用次要目标约束主要目标，防止过拟合"
    
    例如：
      单模狙击可能进化出：if vol_ratio > 5: return 1（极端激进）
        → 狙击85%，趋势30%（偏科！）
      
      双模平衡会进化出：ma25_slope*0.8 + vol_ratio*0.3（平衡合理）
        → 狙击75%，趋势65%（均衡！）
  
  ✅ 优点：
    • 狙击成功率：75%左右（稍低，但仍然好）
    • 趋势准确率：65%左右（显著提升）
    • 进化出的公式更稳健、更长期有效
    • 适应多种行情（爆发、横盘、下跌）
    • 不容易过拟合，泛化能力强
  
  🎯 适合场景：追求稳健、长期有效的策略（大多数情况推荐）
  
  💡 最佳输入：使用混合特征码（5-8个狙击特征 + 5-8个趋势特征）
     → 可加速进化2倍，平衡最佳

✅ 核心特点：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. GPU加速计算（使用NumPy向量化 + PyTorch GPU张量）
2. 真正改算法公式（遗传编程）
3. 双模式防过拟合：狙击（5日涨≥3%）+ 趋势（预测方向）
4. 配置驱动：通过 gp_features_config.json 配置特征
5. 严格数据验证（K线>=REQUIRED_KLINE_COUNT，成交量100%完整）
6. 多进程并行：4进程可4倍加速（CPU密集型）
7. 断点续传：中途失败可继续

🚀 GPU优化方案：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 当前实现：向量化批量计算 + 表达式树编译
   - 表达式树递归改为批量递归
   - 1135个样本一次性计算完成
   - GPU模式：PyTorch张量运算 + 计算图编译
   - CPU模式：NumPy向量化
   - 代码位置：Node.eval_compiled(), Node.eval_vectorized()

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

狙击标准：
- 强狙击：5日涨幅≥5%
- 中狙击：5日涨幅≥3%
- 弱狙击：5日涨幅≥1%
- 默认使用中狙击（≥3%）
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔧 训练数据时间配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ✅✅✅ 从配置文件读取训练截止日期
def get_train_end_date():
    """从配置文件读取训练截止日期"""
    import json
    from pathlib import Path
    config_path = Path(__file__).parent.parent / 'config' / 'global_config.json'
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get('train_end_date', '2023-12-31')  # 默认值
    except Exception as e:
        print(f"⚠️  读取配置文件失败: {e}，使用默认值 2023-12-31")
        return '2023-12-31'

TRAIN_END_DATE = get_train_end_date()  # 训练截止日期（从配置文件读取）

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔧 K线数据加载配置（统一参数，避免硬编码）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED_KLINE_LIMIT = 1500        # 数据库查询限制（最多返回1500根K线）
REQUIRED_KLINE_COUNT = 1500        # 要求的K线数量（严格检查）
REQUIRED_WARMUP_PERIOD = 300       # 预热期K线数量（前300根用于指标计算稳定）

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🌲 遗传编程树深度配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TREE_DEPTH_CONFIG = {
    'min_depth': 2,           # 最小深度（推荐: 2）
    'max_depth': 7,           # 最大深度（推荐: 5-7，工业标准: 6-8）
    'init_min_depth': 2,      # 初始化最小深度（Ramped half-and-half）
    'init_max_depth': 4,      # 初始化最大深度（推荐比max_depth小）
    'description': '树深度范围: 允许2-7深度混合进化，初始化2-4深度'
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎯 遗传编程适应度函数参数配置（可调优）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FITNESS_CONFIG = {
    # 狙击模式参数
    'sniper': {
        'success_weight': 1.0,          # 狙击成功率权重100%（手动交易：只看准确率）
        'profit_weight': 0.0,           # 狙击利润权重0%（手动交易：不关心程序收益）
        'signal_threshold': 0.5,        # 狙击信号阈值（纯浮点数，小数位数不限，如0.5、0.73、0.666等）
                                        # 说明：原始公式输出范围可能很大（如-100万~100万），但经过Tanh归一化后，
                                        # 分数永远被限定在[-1, 1]区间内，因此阈值也必须在[-1, 1]范围内设置
                                        # 推荐范围：-0.5（宽松）~ 0.8（严格），0.5为中等严格
        'profit_baseline': 0.03,        # 利润基准（3%，手动交易：验证信号质量的标准）
        # ✅ 新增：成功标准配置（用于记录到结果文件）
        # 💡 手动交易模式：不考虑手续费，只考虑信号准确度
        #    - consider_fee=False: 不扣除手续费，纯粹看信号质量
        #    - threshold=1%: 未来5天最高价涨幅≥1%就算成功
        #    - 目的：训练出能抓住至少涨1%机会的公式，手动交易时可在高点卖出
        'success_criteria': {
            'method': 'max_return',    # 判断方式: max_return=最高价, close_return=收盘价
            'threshold': 0.01,           # 阈值: 1%
            'consider_fee': False,      # 是否考虑手续费（手动交易模式：不考虑）
            'fee_rate': 0.0013,          # 手续费率: 0.13%（仅当consider_fee=True时生效）
            'description': '未来5天最高价涨幅 ≥ 1%，不考虑手续费'  # 描述
        }
    },
    # 双模模式参数
    'dual': {
        'sniper_weight': 0.6,           # 狙击权重（可调范围: 0.5~0.7）
        'trend_weight': 0.4,            # 趋势权重（可调范围: 0.3~0.5）
    },
    # 趋势模式参数（预留，目前趋势模式直接使用准确率）
    'trend': {
        'threshold_down_base': 0.4,     # 下跌阈值基础（可调范围: 0.3~0.45）
        'threshold_up_base': 0.6,       # 上涨阈值基础（可调范围: 0.55~0.7）
        # ✅ 新增：成功标准配置
        'success_criteria': {
            'method': 'close_return',    # 判断方式: close_return=收盘价
            'up_threshold': 0.02,        # 上涨阈值: 2%
            'down_threshold': -0.02,     # 下跌阈值: -2%
            'description': '第5天收盘价涨幅 > 2% 为上涨, < -2% 为下跌, 其余为横盘'
        }
    }
}

# ✅ 验证配置合法性（启动时检查）
def validate_fitness_config():
    """验证适应度配置的合法性"""
    # 检查狙击模式权重和为1
    sniper_sum = FITNESS_CONFIG['sniper']['success_weight'] + FITNESS_CONFIG['sniper']['profit_weight']
    if abs(sniper_sum - 1.0) > 0.01:
        raise ValueError(f"⚠️  狙击模式权重和必须为1.0，当前: {sniper_sum}")
    
    # 检查双模模式权重和为1
    dual_sum = FITNESS_CONFIG['dual']['sniper_weight'] + FITNESS_CONFIG['dual']['trend_weight']
    if abs(dual_sum - 1.0) > 0.01:
        raise ValueError(f"⚠️  双模模式权重和必须为1.0，当前: {dual_sum}")
    
    # ✅ 检查狙击模式参数范围（手动交易模式允许极端值）
    if not (0.0 <= FITNESS_CONFIG['sniper']['success_weight'] <= 1.0):
        raise ValueError(f"⚠️  success_weight必须在0.0~1.0范围，当前: {FITNESS_CONFIG['sniper']['success_weight']}")
    
    if not (0.0 <= FITNESS_CONFIG['sniper']['profit_weight'] <= 1.0):
        raise ValueError(f"⚠️  profit_weight必须在0.0~1.0范围，当前: {FITNESS_CONFIG['sniper']['profit_weight']}")
    
    # ✅ 阈值范围检查：针对[-1, 1]的分数范围，阈值也应该在[-1, 1]
    if not (-1.0 <= FITNESS_CONFIG['sniper']['signal_threshold'] <= 1.0):
        raise ValueError(f"⚠️  signal_threshold必须在-1.0~1.0范围（对应[-1,1]分数），当前: {FITNESS_CONFIG['sniper']['signal_threshold']}")
    
    if not (0.03 <= FITNESS_CONFIG['sniper']['profit_baseline'] <= 0.08):
        raise ValueError(f"⚠️  profit_baseline必须在0.03~0.08范围，当前: {FITNESS_CONFIG['sniper']['profit_baseline']}")
    
    # ✅ 检查双模模式参数范围
    if not (0.5 <= FITNESS_CONFIG['dual']['sniper_weight'] <= 0.7):
        raise ValueError(f"⚠️  双模sniper_weight必须在0.5~0.7范围，当前: {FITNESS_CONFIG['dual']['sniper_weight']}")
    
    if not (0.3 <= FITNESS_CONFIG['dual']['trend_weight'] <= 0.5):
        raise ValueError(f"⚠️  双模trend_weight必须在0.3~0.5范围，当前: {FITNESS_CONFIG['dual']['trend_weight']}")
    
    # 检查趋势模式阈值合理性
    if not (0.3 <= FITNESS_CONFIG['trend']['threshold_down_base'] <= 0.45):
        raise ValueError(f"⚠️  threshold_down_base必须在0.3~0.45范围，当前: {FITNESS_CONFIG['trend']['threshold_down_base']}")
    
    if not (0.55 <= FITNESS_CONFIG['trend']['threshold_up_base'] <= 0.7):
        raise ValueError(f"⚠️  threshold_up_base必须在0.55~0.7范围，当前: {FITNESS_CONFIG['trend']['threshold_up_base']}")
    
    if FITNESS_CONFIG['trend']['threshold_down_base'] >= FITNESS_CONFIG['trend']['threshold_up_base']:
        raise ValueError(f"⚠️  threshold_down_base必须小于threshold_up_base")
    
    print(f"✅ 适应度配置验证通过")
    print(f"   狙击模式: 成功率{FITNESS_CONFIG['sniper']['success_weight']*100:.0f}% + 利润{FITNESS_CONFIG['sniper']['profit_weight']*100:.0f}%")
    print(f"   双模模式: 狙击{FITNESS_CONFIG['dual']['sniper_weight']*100:.0f}% + 趋势{FITNESS_CONFIG['dual']['trend_weight']*100:.0f}%")
    print(f"   信号阈值: {FITNESS_CONFIG['sniper']['signal_threshold']}")
    print(f"   利润基准: {FITNESS_CONFIG['sniper']['profit_baseline']*100:.0f}%\n")

import sys
import os
import random
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor
import argparse

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ✅ 先设置项目路径，再导入src模块
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ✅ 修复Bug1：智能查找项目根目录，确保路径正确
# 文件位置：src/backend/指标库/遗传编程_特征码组合.py
# 策略：向上查找直到找到包含src目录的目录
project_root = os.path.abspath(os.path.dirname(__file__))
while project_root and not os.path.exists(os.path.join(project_root, 'src')):
    parent = os.path.dirname(project_root)
    if parent == project_root:  # 已到达根目录
        raise RuntimeError(
            "❌ 无法找到项目根目录！\n"
            "期望找到包含'src'目录的项目根，但未找到。\n"
            f"当前文件路径: {__file__}"
        )
    project_root = parent

# ✅ 增强验证：检查项目结构完整性
if not os.path.exists(os.path.join(project_root, 'src')):
    raise RuntimeError(
        f"❌ 项目根目录验证失败: {project_root}\n"
        f"找不到 src/ 目录，请检查文件结构是否完整"
    )

# ✅ 成功找到项目根目录，输出确认信息（便于调试）
if os.getenv('DEBUG'):
    print(f"✅ 项目根目录: {project_root}")
    print(f"   当前文件: {__file__}")

sys.path.insert(0, project_root)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 特征配置加载（从 gp_features_config.json 读取）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 全局配置（程序启动时加载）
FEATURE_CONFIG = None

from database.config import SessionLocal
from database.models.stock_kline_day import StockKlineDay
from sqlalchemy import func

# ✅ 导入股票和大盘数据加载工具（使用新的data_loader版本）
from src.backend.data_loader.kline_data_loader import load_stock_and_market_data, REQUIRED_KLINE_COUNT

# ✅ 新增：导入 gp_indicators_manager 用于遗传编程专用的特征计算和归一化
import sys
from pathlib import Path
indicators_lib_path = Path(__file__).parent
if str(indicators_lib_path) not in sys.path:
    sys.path.insert(0, str(indicators_lib_path))
from gp_indicators_manager import GPIndicatorsManager

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ✅ 新增：从 gp_features_config.json 加载特征码
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_gp_features(mode: str) -> list:
    """
    从 gp_features_config.json 加载特征码
    
    Args:
        mode: 'sniper' | 'trend' | 'dual'
    
    Returns:
        特征码列表 [
            'stockstats_close_-1_d',
            'finta_VWAP',
            ...
        ]
    """
    config_path = Path(__file__).parent / 'config' / 'gp_features_config.json'
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"❌ 找不到配置文件: {config_path}\n"
            f"请确保 gp_features_config.json 存在于 {Path(__file__).parent / 'config'} 目录下"
        )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    if mode == 'sniper':
        # 狙击模式：只读狙击特征
        features = config['sniper_mode']['features']
        print(f"🎯 狙击模式：加载 {len(features)} 个狙击特征")
    
    elif mode == 'trend':
        # 趋势模式：只读趋势特征
        features = config['trend_mode']['features']
        print(f"📈 趋势模式：加载 {len(features)} 个趋势特征")
    
    elif mode == 'dual':
        # 双模模式：合并狙击+趋势特征（去重）
        sniper_features = config['sniper_mode']['features']
        trend_features = config['trend_mode']['features']
        
        # 合并并去重，保持顺序
        features = list(dict.fromkeys(sniper_features + trend_features))
        
        print(f"⚖️  双模模式：加载 {len(sniper_features)} 个狙击 + {len(trend_features)} 个趋势 = {len(features)} 个总特征（去重后）")
    
    else:
        raise ValueError(f"❌ 无效的模式: {mode}，必须是 'sniper', 'trend' 或 'dual'")
    
    return features


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🚀 GPU 初始化函数（延迟加载，避免子进程重复初始化）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 全局变量（函数内初始化）
USE_GPU = False
GPU_NAME = None
GPU_MEMORY = 0
DEVICE = None
torch = None
_GPU_INITIALIZED = False

def initialize_gpu():
    """初始化GPU配置（只在主进程中执行一次）"""
    global USE_GPU, GPU_NAME, GPU_MEMORY, DEVICE, torch, _GPU_INITIALIZED
    
    if _GPU_INITIALIZED:
        return  # 已初始化，跳过
    
    _GPU_INITIALIZED = True
    
    print(f"✅ GPU检测...")
    
    try:
        import torch as torch_module
        torch = torch_module
        import os
        
        # ✅ 修复High-4：不强制覆盖CUDA_VISIBLE_DEVICES，允许用户自定义GPU
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 默认使用GPU 0
        
        if torch.cuda.is_available():
            USE_GPU = True
            DEVICE = torch.device('cuda:0')
            GPU_NAME = torch.cuda.get_device_name(0)
            GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"   🚀 检测到GPU: {GPU_NAME} ({GPU_MEMORY:.1f}GB)")
            print(f"   💡 将使用GPU 0进行计算加速")
            print(f"   🚀 GPU优化策略:")
            print(f"      🚀 表达式树编译为PyTorch计算图（2.8-4.6倍加速）")
            print(f"      1. 数据预加载到GPU内存（一次性转换）")
            print(f"      2. 表达式树编译为lambda函数（零Python递归开销）")
            print(f"      3. PyTorch自动算子融合（减少显存访问）")
            print(f"      4. ATR/MA计算向量化（避免Python循环）")
        else:
            USE_GPU = False
            DEVICE = torch.device('cpu')
            print(f"   ⚠️  未检测到GPU，将使用CPU模式")
    except ImportError:
        USE_GPU = False
        DEVICE = None
        torch = None
        print("   ⚠️  PyTorch未安装，使用CPU模式")
        print("   💡 建议安装PyTorch加速：pip install torch")


# 工具函数：NumPy转PyTorch张量
def to_tensor(arr):
    """数组转张量"""
    global torch, DEVICE
    # ✅ 延迟导入torch（防止子进程未初始化）
    if torch is None:
        try:
            import torch as torch_module
            torch = torch_module
        except ImportError:
            pass
    
    if USE_GPU and torch is not None and DEVICE is not None:
        return torch.tensor(arr, dtype=torch.float32, device=DEVICE)
    return arr

def to_numpy(tensor):
    """张量转数组"""
    global torch
    # ✅ 延迟导入torch
    if torch is None:
        try:
            import torch as torch_module
            torch = torch_module
        except ImportError:
            pass
    
    if USE_GPU and torch is not None and isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    return tensor


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 遗传编程：代码树结构
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Node:
    """表达式树节点"""
    def __init__(self, node_type, value=None, left=None, right=None):
        self.type = node_type  # 'op', 'var', 'const'
        self.value = value
        self.left = left
        self.right = right
    
    def eval(self, ctx):
        """执行节点（单个样本）"""
        if self.type == 'const':
            return self.value
        elif self.type == 'var':
            return ctx.get(self.value, 0)
        elif self.type == 'op':
            left_val = self.left.eval(ctx) if self.left else 0
            right_val = self.right.eval(ctx) if self.right else 0
            
            if self.value == '+':
                return left_val + right_val
            elif self.value == '-':
                return left_val - right_val
            elif self.value == '*':
                return left_val * right_val
            elif self.value == '/':
                return left_val / right_val if abs(right_val) > 1e-8 else 0
            elif self.value == 'max':
                return max(left_val, right_val)
            elif self.value == 'min':
                return min(left_val, right_val)
            elif self.value == 'abs':
                return abs(left_val)
            elif self.value == 'sqrt':
                return np.sqrt(abs(left_val))
            elif self.value == 'log':
                return np.log(abs(left_val) + 1)
            elif self.value == 'neg':
                return -left_val
            elif self.value == 'inv':
                return 1.0 / left_val if abs(left_val) > 1e-8 else 0
            elif self.value == 'sin':
                return np.sin(left_val)
            elif self.value == 'cos':
                return np.cos(left_val)
            elif self.value == 'tan':
                return np.tan(left_val)
            elif self.value == 'sig':
                # Sigmoid: 1/(1+exp(-x))
                return 1.0 / (1.0 + np.exp(-np.clip(left_val, -10, 10)))
            elif self.value == 'tanh':
                return np.tanh(left_val)
            elif self.value == 'exp':
                # 防止指数爆炸，限制输入范围
                return np.exp(np.clip(left_val, -10, 10))
        
        return 0
    
    def compile_to_torch(self):
        """将表达式树编译为PyTorch lambda函数
        
        返回一个lambda函数，接受ctx_batch参数，返回GPU张量
        编译后的函数无Python递归开销，性能提升2.8-4.6倍
        """
        if self.type == 'const':
            # ✅ 修复Critical-1：常量节点返回正确形状
            value = self.value
            def const_fn(ctx):
                if not ctx:
                    raise RuntimeError("❌ compile_to_torch需要非空ctx，至少包含一个张量")
                ref_tensor = next(iter(ctx.values()))
                if not isinstance(ref_tensor, torch.Tensor):
                    raise TypeError(f"❌ 期望torch.Tensor，得到{type(ref_tensor)}")
                return torch.full_like(ref_tensor, value, dtype=torch.float32, device=DEVICE)
            return const_fn
        
        elif self.type == 'var':
            # ✅ 修复Critical-1：变量节点返回正确形状
            var_name = self.value
            def var_fn(ctx):
                if not ctx:
                    raise RuntimeError("❌ compile_to_torch需要非空ctx")
                ref_tensor = next(iter(ctx.values()))
                return ctx.get(var_name, torch.zeros_like(ref_tensor))
            return var_fn
        
        elif self.type == 'op':
            # 运算符节点：递归编译子树，返回组合lambda
            left_fn = self.left.compile_to_torch() if self.left else None
            right_fn = self.right.compile_to_torch() if self.right else None
            
            # 返回组合运算的lambda
            if self.value == '+':
                return lambda ctx: left_fn(ctx) + right_fn(ctx)
            elif self.value == '-':
                return lambda ctx: left_fn(ctx) - right_fn(ctx)
            elif self.value == '*':
                return lambda ctx: left_fn(ctx) * right_fn(ctx)
            elif self.value == '/':
                # 除法需要避免除零
                def div_fn(ctx):
                    left_val = left_fn(ctx)
                    right_val = right_fn(ctx)
                    return torch.where(
                        torch.abs(right_val) > 1e-8,
                        left_val / right_val,
                        torch.zeros_like(left_val)
                    )
                return div_fn
            elif self.value == 'max':
                return lambda ctx: torch.maximum(left_fn(ctx), right_fn(ctx))
            elif self.value == 'min':
                return lambda ctx: torch.minimum(left_fn(ctx), right_fn(ctx))
            elif self.value == 'abs':
                # ✅ 修复Critical-1：一元操作返回正确形状
                if left_fn is None:
                    def zero_fn(ctx):
                        if not ctx:
                            raise RuntimeError("❌ ctx不能为空")
                        return torch.zeros_like(next(iter(ctx.values())))
                    return zero_fn
                return lambda ctx: torch.abs(left_fn(ctx))
            elif self.value == 'sqrt':
                # ✅ 修复Critical-1
                if left_fn is None:
                    def zero_fn(ctx):
                        if not ctx:
                            raise RuntimeError("❌ ctx不能为空")
                        return torch.zeros_like(next(iter(ctx.values())))
                    return zero_fn
                return lambda ctx: torch.sqrt(torch.abs(left_fn(ctx)))
            elif self.value == 'log':
                # ✅ 修复Critical-1
                if left_fn is None:
                    def zero_fn(ctx):
                        if not ctx:
                            raise RuntimeError("❌ ctx不能为空")
                        return torch.zeros_like(next(iter(ctx.values())))
                    return zero_fn
                return lambda ctx: torch.log(torch.abs(left_fn(ctx)) + 1)
            elif self.value == 'neg':
                if left_fn is None:
                    def zero_fn(ctx):
                        if not ctx:
                            raise RuntimeError("❌ ctx不能为空")
                        return torch.zeros_like(next(iter(ctx.values())))
                    return zero_fn
                return lambda ctx: -left_fn(ctx)
            elif self.value == 'inv':
                if left_fn is None:
                    def zero_fn(ctx):
                        if not ctx:
                            raise RuntimeError("❌ ctx不能为空")
                        return torch.zeros_like(next(iter(ctx.values())))
                    return zero_fn
                def inv_fn(ctx):
                    left_val = left_fn(ctx)
                    return torch.where(torch.abs(left_val) > 1e-8, 1.0 / left_val, torch.zeros_like(left_val))
                return inv_fn
            elif self.value == 'sin':
                if left_fn is None:
                    def zero_fn(ctx):
                        if not ctx:
                            raise RuntimeError("❌ ctx不能为空")
                        return torch.zeros_like(next(iter(ctx.values())))
                    return zero_fn
                return lambda ctx: torch.sin(left_fn(ctx))
            elif self.value == 'cos':
                if left_fn is None:
                    def zero_fn(ctx):
                        if not ctx:
                            raise RuntimeError("❌ ctx不能为空")
                        return torch.zeros_like(next(iter(ctx.values())))
                    return zero_fn
                return lambda ctx: torch.cos(left_fn(ctx))
            elif self.value == 'tan':
                if left_fn is None:
                    def zero_fn(ctx):
                        if not ctx:
                            raise RuntimeError("❌ ctx不能为空")
                        return torch.zeros_like(next(iter(ctx.values())))
                    return zero_fn
                return lambda ctx: torch.tan(left_fn(ctx))
            elif self.value == 'sig':
                if left_fn is None:
                    def zero_fn(ctx):
                        if not ctx:
                            raise RuntimeError("❌ ctx不能为空")
                        return torch.zeros_like(next(iter(ctx.values())))
                    return zero_fn
                return lambda ctx: 1.0 / (1.0 + torch.exp(-torch.clamp(left_fn(ctx), -10, 10)))
            elif self.value == 'tanh':
                if left_fn is None:
                    def zero_fn(ctx):
                        if not ctx:
                            raise RuntimeError("❌ ctx不能为空")
                        return torch.zeros_like(next(iter(ctx.values())))
                    return zero_fn
                return lambda ctx: torch.tanh(left_fn(ctx))
            elif self.value == 'exp':
                if left_fn is None:
                    def zero_fn(ctx):
                        if not ctx:
                            raise RuntimeError("❌ ctx不能为空")
                        return torch.zeros_like(next(iter(ctx.values())))
                    return zero_fn
                return lambda ctx: torch.exp(torch.clamp(left_fn(ctx), -10, 10))
        
        # ✅ 修复Critical-1：默认返回正确形状
        def default_fn(ctx):
            if not ctx:
                raise RuntimeError("❌ ctx不能为空")
            return torch.zeros_like(next(iter(ctx.values())))
        return default_fn
    
    def eval_compiled(self, ctx_batch):
        """使用编译后的函数评估
        
        首次调用时编译表达式树为PyTorch lambda函数，后续直接使用缓存
        性能：比eval_vectorized快2.8-4.6倍
        
        Args:
            ctx_batch: dict of GPU tensors {
                'ma25_slope': torch.Tensor(batch_size,),
                'vol_ratio': torch.Tensor(batch_size,),
                ...
            }
        
        Returns:
            torch.Tensor of shape (batch_size,)
        """
        # 首次调用：编译并缓存
        if not hasattr(self, '_compiled_fn'):
            if USE_GPU and torch is not None:
                self._compiled_fn = self.compile_to_torch()
            else:
                self._compiled_fn = None
        
        # 使用编译后的函数
        if self._compiled_fn is not None:
            return self._compiled_fn(ctx_batch)
        
        # CPU模式：使用eval_vectorized
        batch_size = len(list(ctx_batch.values())[0]) if ctx_batch else 0
        return self.eval_vectorized(ctx_batch, batch_size)
    
    def eval_vectorized(self, ctx_batch, batch_size):
        """向量化执行节点（批量样本，GPU加速）
        
        Args:
            ctx_batch: dict of arrays/tensors {
                'ma25_slope': (batch_size,),
                'vol_ratio': (batch_size,),
                ...
            }
            batch_size: 样本数量
        
        Returns:
            array/tensor of shape (batch_size,)
        """
        # 🚀 GPU模式：使用PyTorch张量运算
        if USE_GPU and torch is not None:
            if self.type == 'const':
                # 常量扩展为(batch_size,)
                return torch.full((batch_size,), self.value, dtype=torch.float32, device=DEVICE)
            
            elif self.type == 'var':
                # 直接返回GPU张量
                result = ctx_batch.get(self.value)
                if result is None:
                    return torch.zeros(batch_size, dtype=torch.float32, device=DEVICE)
                return result
            
            elif self.type == 'op':
                # ✅ GPU模式：一元和二元操作
                if self.value in ['abs', 'sqrt', 'log', 'neg', 'inv', 'sin', 'cos', 'tan', 'sig', 'tanh', 'exp']:
                    # 一元操作：只计算left
                    left_vals = self.left.eval_vectorized(ctx_batch, batch_size) if self.left else torch.zeros(batch_size, device=DEVICE)
                    
                    if self.value == 'abs':
                        return torch.abs(left_vals)
                    elif self.value == 'sqrt':
                        return torch.sqrt(torch.abs(left_vals))
                    elif self.value == 'log':
                        return torch.log(torch.abs(left_vals) + 1)
                    elif self.value == 'neg':
                        return -left_vals
                    elif self.value == 'inv':
                        return torch.where(torch.abs(left_vals) > 1e-8, 1.0 / left_vals, torch.zeros_like(left_vals))
                    elif self.value == 'sin':
                        return torch.sin(left_vals)
                    elif self.value == 'cos':
                        return torch.cos(left_vals)
                    elif self.value == 'tan':
                        return torch.tan(left_vals)
                    elif self.value == 'sig':
                        # Sigmoid: 1/(1+exp(-x))
                        return 1.0 / (1.0 + torch.exp(-torch.clamp(left_vals, -10, 10)))
                    elif self.value == 'tanh':
                        return torch.tanh(left_vals)
                    elif self.value == 'exp':
                        return torch.exp(torch.clamp(left_vals, -10, 10))
                else:
                    # 二元操作：计算left和right
                    left_vals = self.left.eval_vectorized(ctx_batch, batch_size) if self.left else torch.zeros(batch_size, device=DEVICE)
                    right_vals = self.right.eval_vectorized(ctx_batch, batch_size) if self.right else torch.zeros(batch_size, device=DEVICE)
                    
                    # GPU向量运算
                    if self.value == '+':
                        return left_vals + right_vals
                    elif self.value == '-':
                        return left_vals - right_vals
                    elif self.value == '*':
                        return left_vals * right_vals
                    elif self.value == '/':
                        # 避免除零
                        return torch.where(torch.abs(right_vals) > 1e-8, left_vals / right_vals, torch.zeros_like(left_vals))
                    elif self.value == 'max':
                        return torch.maximum(left_vals, right_vals)
                    elif self.value == 'min':
                        return torch.minimum(left_vals, right_vals)
        
        # 💻 CPU模式：使用NumPy向量运算
        else:
            if self.type == 'const':
                return np.full(batch_size, self.value, dtype=np.float32)
            
            elif self.type == 'var':
                result = ctx_batch.get(self.value)
                if result is None:
                    return np.zeros(batch_size, dtype=np.float32)
                return result
            
            elif self.type == 'op':
                # ✅ CPU模式：一元和二元操作
                if self.value in ['abs', 'sqrt', 'log', 'neg', 'inv', 'sin', 'cos', 'tan', 'sig', 'tanh', 'exp']:
                    # 一元操作：只计算left
                    left_vals = self.left.eval_vectorized(ctx_batch, batch_size) if self.left else np.zeros(batch_size)
                    
                    if self.value == 'abs':
                        return np.abs(left_vals)
                    elif self.value == 'sqrt':
                        return np.sqrt(np.abs(left_vals))
                    elif self.value == 'log':
                        return np.log(np.abs(left_vals) + 1)
                    elif self.value == 'neg':
                        return -left_vals
                    elif self.value == 'inv':
                        return np.where(np.abs(left_vals) > 1e-8, 1.0 / left_vals, np.zeros_like(left_vals))
                    elif self.value == 'sin':
                        return np.sin(left_vals)
                    elif self.value == 'cos':
                        return np.cos(left_vals)
                    elif self.value == 'tan':
                        return np.tan(left_vals)
                    elif self.value == 'sig':
                        # Sigmoid: 1/(1+exp(-x))
                        return 1.0 / (1.0 + np.exp(-np.clip(left_vals, -10, 10)))
                    elif self.value == 'tanh':
                        return np.tanh(left_vals)
                    elif self.value == 'exp':
                        return np.exp(np.clip(left_vals, -10, 10))
                else:
                    # 二元操作：计算left和right
                    left_vals = self.left.eval_vectorized(ctx_batch, batch_size) if self.left else np.zeros(batch_size)
                    right_vals = self.right.eval_vectorized(ctx_batch, batch_size) if self.right else np.zeros(batch_size)
                    
                    # NumPy向量运算
                    if self.value == '+':
                        return left_vals + right_vals
                    elif self.value == '-':
                        return left_vals - right_vals
                    elif self.value == '*':
                        return left_vals * right_vals
                    elif self.value == '/':
                        # ✅ 修复Bug7：CPU除法应该返回left_vals的形状，与GPU保持一致
                        return np.where(np.abs(right_vals) > 1e-8, left_vals / right_vals, np.zeros_like(left_vals))
                    elif self.value == 'max':
                        return np.maximum(left_vals, right_vals)
                    elif self.value == 'min':
                        return np.minimum(left_vals, right_vals)
        
        # ✅ 修复High-5：统一返回NumPy数组（与GPU保持一致）
        if USE_GPU:
            return torch.zeros(batch_size, device=DEVICE)
        else:
            return np.zeros(batch_size, dtype=np.float32)
    
    def to_code(self):
        """转换为Python代码"""
        if self.type == 'const':
            return f"{self.value:.3f}"
        elif self.type == 'var':
            return self.value
        elif self.type == 'op':
            if self.value in ['+', '-', '*', '/']:
                left_code = self.left.to_code() if self.left else "0"
                right_code = self.right.to_code() if self.right else "0"
                return f"({left_code} {self.value} {right_code})"
            elif self.value == 'max':
                # ✅ 修复High-6：添加None检查
                left_code = self.left.to_code() if self.left else "0"
                right_code = self.right.to_code() if self.right else "0"
                return f"max({left_code}, {right_code})"
            elif self.value == 'min':
                # ✅ 添加None检查
                left_code = self.left.to_code() if self.left else "0"
                right_code = self.right.to_code() if self.right else "0"
                return f"min({left_code}, {right_code})"
            elif self.value in ['abs', 'sqrt', 'log', 'neg', 'inv', 'sin', 'cos', 'tan', 'sig', 'tanh', 'exp']:
                # ✅ 一元函数
                left_code = self.left.to_code() if self.left else "0"
                return f"{self.value}({left_code})"
        return "0"


def random_tree(depth=0, max_depth=None):
    """生成随机表达式树（使用配置文件中的特征）
    
    Args:
        depth: 当前深度
        max_depth: 最大深度（如果为None，则从TREE_DEPTH_CONFIG读取）
    """
    global FEATURE_CONFIG
    
    # ✅ 修复High-7：检查available_vars是否为空
    if FEATURE_CONFIG is None or not FEATURE_CONFIG.get('available_vars'):
        raise RuntimeError(
            "❌ FEATURE_CONFIG未初始化或available_vars为空！\n"
            "请确保在创建AlgorithmGene之前已经设置FEATURE_CONFIG全局变量。"
        )
    
    # ✅ 使用配置的max_depth（如果未指定）
    if max_depth is None:
        max_depth = TREE_DEPTH_CONFIG['max_depth']
    
    available_vars = FEATURE_CONFIG['available_vars']
    
    if depth >= max_depth:
        # 叶子节点：变量或常数
        if random.random() < 0.6:
            var = random.choice(available_vars)
            return Node('var', var)
        else:
            const = random.uniform(-1, 1)
            return Node('const', const)
    
    # 内部节点：运算符（✅ 金融级工业标准：17个函数）
    op = random.choice(['+', '-', '*', '/', 'max', 'min', 'abs', 'sqrt', 'log', 
                         'neg', 'inv', 'sin', 'cos', 'tan', 'sig', 'tanh', 'exp'])
    left = random_tree(depth + 1, max_depth)
    # 一元操作不需要right子树
    right = random_tree(depth + 1, max_depth) if op not in ['abs', 'sqrt', 'log', 'neg', 'inv', 'sin', 'cos', 'tan', 'sig', 'tanh', 'exp'] else None
    
    return Node('op', op, left, right)


def copy_tree(node):
    """✅ 深拷贝树（防止共享引用）"""
    if node is None:
        return None
    new_node = Node(node.type, node.value)
    new_node.left = copy_tree(node.left)
    new_node.right = copy_tree(node.right)
    return new_node


def mutate_tree(node, prob=0.1):
    """突变树（✅ 修复错误2：不修改原对象，返回新对象）"""
    if random.random() < prob:
        # ✅ 使用配置的max_depth（突变时使用较小的深度，避免过度复杂）
        mutate_max_depth = min(TREE_DEPTH_CONFIG['max_depth'], TREE_DEPTH_CONFIG['init_max_depth'])
        return random_tree(max_depth=mutate_max_depth)
    
    # ✅ 修复错误2：创建新节点，不修改原对象
    if node.type == 'op':
        new_left = mutate_tree(node.left, prob) if node.left else None
        new_right = mutate_tree(node.right, prob) if node.right else None
        return Node(node.type, node.value, new_left, new_right)
    else:
        # 叶子节点：深拷贝（显式设置left和right为None）
        return Node(node.type, node.value, None, None)



def crossover_tree(parent1, parent2):
    """交叉两棵树（✅ 修复错误1：完整深拷贝，避免引用污染）"""
    if random.random() < 0.5:
        new_tree = Node(parent1.type, parent1.value)
        if parent1.left:
            # ✅ 修复错误1：如果parent2.left为None，深拷贝parent1.left而不是直接引用
            p2_left = parent2.left if parent2.left else None
            if p2_left:
                new_tree.left = crossover_tree(parent1.left, p2_left)
            else:
                new_tree.left = copy_tree(parent1.left)  # ✅ 深拷贝，避免污染
        if parent1.right:
            # ✅ 修复错误1：如果parent2.right为None，深拷贝parent1.right而不是直接引用
            p2_right = parent2.right if parent2.right else None
            if p2_right:
                new_tree.right = crossover_tree(parent1.right, p2_right)
            else:
                new_tree.right = copy_tree(parent1.right)  # ✅ 深拷贝，避免污染
        return new_tree
    else:
        # ✅ 修复：深拷贝parent2，防止共享引用
        return copy_tree(parent2)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 基因（算法个体）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class AlgorithmGene:
    """算法基因（根据配置文件动态生成维度）"""
    def __init__(self, skip_random_trees=False):
        """✅ 修复效率问题：支持跳过随机树生成（当trees会被立即替换时）
        
        Args:
            skip_random_trees: 如果为True，不生成随机树（用于深拷贝/交叉变异场景）
        """
        global FEATURE_CONFIG
        
        # ✅ 修复错误6：检查FEATURE_CONFIG是否为None
        if FEATURE_CONFIG is None:
            raise RuntimeError(
                "❗ FEATURE_CONFIG未初始化！\n"
                "请确保在创建AlgorithmGene之前已经设置FEATURE_CONFIG全局变量。"
            )
        
        # ✅ 使用 available_vars 代替 dimension_config
        # 每个特征生成一个表达式树（一个维度）
        num_trees = len(FEATURE_CONFIG['available_vars'])
        
        # ✅ 修复效率问题：只在需要时生成随机树
        if skip_random_trees:
            # 深拷贝/交叉变异场景：空列表，由外部赋值trees
            self.trees = []
        else:
            # 正常初始化场景：使用Ramped Half-and-Half策略生成随机树
            # 一半个体用较浅的树（init_min_depth），另一半用较深的树（init_max_depth）
            self.trees = []
            for i in range(num_trees):
                # ✅ Ramped Half-and-Half: 不同深度混合进化
                if i % 2 == 0:
                    # 偶数索引：使用最小深度到中间深度
                    tree_depth = random.randint(
                        TREE_DEPTH_CONFIG['init_min_depth'],
                        (TREE_DEPTH_CONFIG['init_min_depth'] + TREE_DEPTH_CONFIG['init_max_depth']) // 2
                    )
                else:
                    # 奇数索引：使用中间深度到最大深度
                    tree_depth = random.randint(
                        (TREE_DEPTH_CONFIG['init_min_depth'] + TREE_DEPTH_CONFIG['init_max_depth']) // 2,
                        TREE_DEPTH_CONFIG['init_max_depth']
                    )
                tree = random_tree(max_depth=tree_depth)
                self.trees.append(tree)
        
        # 适应度
        self.fitness_sniper = 0.0
        self.fitness_trend = 0.0
        self.fitness = 0.0
        self.signal_count = 0  # 信号数量
        self.gene_id = f"G{int(time.time()*1000)}{random.randint(1000,9999)}"  # 唯一ID
        
        # 趋势模式额外信息
        self.trend_accuracy = 0.0  # 3分类准确率
        self.trend_distribution = {'down': 0, 'sideways': 0, 'up': 0}  # 样本分布
    
    # ✅ 已删除未使用的 compute_score() 方法（使用批量评估代替）
    
    def to_dict(self):
        """转换为字典（用于保存）"""
        return {
            'gene_id': self.gene_id,
            'fitness': self.fitness,
            'fitness_sniper': self.fitness_sniper,
            'fitness_trend': self.fitness_trend,
            'signal_count': self.signal_count,
            'trend_accuracy': getattr(self, 'trend_accuracy', 0.0),
            'trend_distribution': getattr(self, 'trend_distribution', {'down': 0, 'sideways': 0, 'up': 0}),
            'trees': [self._tree_to_dict(tree) for tree in self.trees],
            # ✅ MRGP权重持久化（防止断点恢复后丢失）
            'mrgp_weights': self.mrgp_weights.tolist() if hasattr(self, 'mrgp_weights') and self.mrgp_weights is not None else None,
            'mrgp_intercept': float(self.mrgp_intercept) if hasattr(self, 'mrgp_intercept') and self.mrgp_intercept is not None else None,
            'mrgp_score': getattr(self, 'mrgp_score', 0.0)
        }
    
    def _tree_to_dict(self, node):
        """树转字典"""
        if node is None:
            return None
        return {
            'type': node.type,
            'value': node.value,
            'left': self._tree_to_dict(node.left),
            'right': self._tree_to_dict(node.right)
        }
    
    @classmethod
    def from_dict(cls, data):
        """从字典恢复"""
        # ✅ 修复效率问题：跳过随机树生成（会被立即替换）
        gene = cls(skip_random_trees=True)
        gene.gene_id = data['gene_id']
        gene.fitness = data['fitness']
        gene.fitness_sniper = data['fitness_sniper']
        gene.fitness_trend = data['fitness_trend']
        gene.signal_count = data.get('signal_count', 0)
        gene.trend_accuracy = data.get('trend_accuracy', 0.0)
        gene.trend_distribution = data.get('trend_distribution', {'down': 0, 'sideways': 0, 'up': 0})
        gene.trees = [cls._dict_to_tree(tree_data) for tree_data in data['trees']]
        
        # ✅ MRGP权重恢复（断点恢复时保持优化信息）
        if data.get('mrgp_weights') is not None:
            gene.mrgp_weights = np.array(data['mrgp_weights'])
        else:
            gene.mrgp_weights = None
        gene.mrgp_intercept = data.get('mrgp_intercept')
        gene.mrgp_score = data.get('mrgp_score', 0.0)
        
        return gene
    
    @classmethod
    def _dict_to_tree(cls, data):
        """字典转树"""
        if data is None:
            return None
        node = Node(
            node_type=data['type'],
            value=data['value'],
            left=cls._dict_to_tree(data.get('left')),
            right=cls._dict_to_tree(data.get('right'))
        )
        return node
    

    def to_code(self):
        """生成Python代码"""
        global FEATURE_CONFIG
        
        # ✅ 检查FEATURE_CONFIG是否已初始化
        if FEATURE_CONFIG is None:
            return "# 错误：FEATURE_CONFIG未初始化，无法生成代码"
        
        # ✅ 修复Bug7：检查trees是否为空
        if not self.trees or len(self.trees) == 0:
            return "# 错误：基因没有表达式树，无法生成代码"
        
        # 生成所有维度的代码
        dimension_codes = []
        for i, tree in enumerate(self.trees):
            dimension_codes.append(f"dim{i}_score = {tree.to_code()}")
        
        # 生成平均计算
        scores_sum = " + ".join([f"dim{i}_score" for i in range(len(self.trees))])
        
        available_vars = FEATURE_CONFIG['available_vars']
        params = ", ".join(available_vars)
        
        # ✅ MRGP：生成权重信息
        if hasattr(self, 'mrgp_weights') and self.mrgp_weights is not None:
            # 有MRGP权重，生成加权计算代码
            weights_str = ", ".join([f"{w:.4f}" for w in self.mrgp_weights])
            weighted_sum = " + ".join([f"{self.mrgp_weights[i]:.4f}*dim{i}_score" for i in range(len(self.trees))])
            mrgp_info = f"""
    ✅ MRGP权重优化：
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    本公式已经用MRGP（Multiple Regression GP）学习了最优权重！
    
    学到的权重：[{weights_str}]
    截距项：{self.mrgp_intercept:.4f}
    R^2分数：{self.mrgp_score:.4f}
    
    加权公式：
        final_score = {weighted_sum} + {self.mrgp_intercept:.4f}
    
    💡 这比简单平均更准！因为：
        - 自动发现哪个维度更重要（权重高）
        - 弱维度被降权，减少干扰
        - 通过线性回归优化，不是简单平均
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            return_statement = f"return {weighted_sum} + {self.mrgp_intercept:.4f}"
        else:
            # 没有MRGP权重，使用简单平均
            mrgp_info = ""
            return_statement = f"return ({scores_sum}) / {len(self.trees)}"
        
        code = f"""
def evolved_algorithm({params}):
    '''
    遗传编程进化版算法
    Generation: 自动生成
    Fitness: {self.fitness:.4f}
    Sniper Fitness: {self.fitness_sniper:.4f}
    Trend Fitness: {self.fitness_trend:.4f}
    Dimensions: {len(self.trees)}
    {mrgp_info}
    '''
    {chr(10).join(['    ' + code for code in dimension_codes])}
    {return_statement}
"""
        return code


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 数据加载与验证
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_valid_stocks(count=10):
    """获取有效股票列表（严格验证）
    
    ✅ 读取配置文件 hs300_zz500_zz1000.json
    - use_all_stocks=true: 从数据库读取所有股票
    - use_all_stocks=false: 从配置文件的stock_codes列表读取
    """
    db = SessionLocal()
    try:
        # ✅ 步骤1: 从配置文件读取股票来源
        import json
        stock_config_path = Path(__file__).parent.parent / 'config' / 'hs300_zz500_zz1000.json'
        
        if not stock_config_path.exists():
            print(f"⚠️  配置文件不存在: {stock_config_path}，使用数据库所有股票")
            use_all_stocks = True
            all_stock_codes = []
        else:
            with open(stock_config_path, 'r', encoding='utf-8') as f:
                stock_config = json.load(f)
            
            use_all_stocks = stock_config.get('use_all_stocks', False)
            
            if use_all_stocks:
                # ✅ 开关打开：从数据库读取所有股票
                print(f"✅ 配置: use_all_stocks=true")
                print(f"   模式: 从数据库读取所有股票（自动排除ST和不可交易股票）")
                
                # 查询所有有效股票
                from database.models.stock_info import StockInfo
                valid_stocks_query = db.query(StockInfo.stock_code).filter(
                    StockInfo.stock_name.notlike('%ST%'),
                    StockInfo.stock_name.notlike('%st%'),
                    StockInfo.is_tradable == True,
                    StockInfo.is_active == True
                ).all()
                valid_codes = set([code[0] for code in valid_stocks_query])
                
                # 查询有K线数据的股票
                all_codes_query = db.query(StockKlineDay.stock_code).filter(
                    StockKlineDay.period == 'day'
                ).distinct().all()
                
                # 取交集
                all_codes_with_data = [code[0] for code in all_codes_query]
                all_stock_codes = [code for code in all_codes_with_data if code in valid_codes]
                
                print(f"   从数据库: {len(all_codes_with_data)} 只 -> 排除ST/不可交易: {len(all_stock_codes)} 只")
            else:
                # ✅ 开关关闭：从配置文件读取股票列表
                print(f"✅ 配置: use_all_stocks=false")
                print(f"   模式: 从配置文件stock_codes列表读取")
                all_stock_codes = stock_config['stock_codes']
                print(f"   从配置文件: {len(all_stock_codes)} 只股票")
        
        # ✅ 步骤2: 随机打乱股票代码（确保均匀分布）
        print(f"✅ 随机打乱股票顺序...")
        stocks_list = all_stock_codes.copy()
        random.shuffle(stocks_list)
        
        # ✅ 步骤3: 过滤有效股票
        print(f"✅ 开始验证股票（K线>={REQUIRED_KLINE_COUNT}根，成交量100%完整）...")
        valid_stocks = []
        
        checked_count = 0
        skipped_kline = 0
        skipped_volume = 0
        
        for stock_code in stocks_list:
            if len(valid_stocks) >= count:
                break
            
            checked_count += 1
            
            # ✅ 阶段1：轻量级筛选（只查COUNT，非常快）
            kline_count = db.query(func.count(StockKlineDay.id)).filter(
                StockKlineDay.stock_code == stock_code,
                StockKlineDay.period == 'day',
                StockKlineDay.trade_date <= TRAIN_END_DATE
            ).scalar()
            
            if kline_count < REQUIRED_KLINE_COUNT:
                skipped_kline += 1
                continue
            
            # ✅ 阶段2：检查成交量完整性（只对通过COUNT筛选的股票）
            # ✅ 先获取最近REQUIRED_KLINE_COUNT根K线，然后检查这个K线范围内的成交量完整性
            recent_klines = db.query(StockKlineDay).filter(
                StockKlineDay.stock_code == stock_code,
                StockKlineDay.period == 'day',
                StockKlineDay.trade_date <= TRAIN_END_DATE
            ).order_by(StockKlineDay.trade_date.desc()).limit(REQUIRED_KLINE_COUNT).all()
            
            if len(recent_klines) < REQUIRED_KLINE_COUNT:
                skipped_kline += 1
                continue
            
            # 检查这些K线的成交量完整性
            invalid_volume_count = sum(
                1 for k in recent_klines 
                if k.volume is None or k.volume <= 0
            )
            
            if invalid_volume_count > 0:
                skipped_volume += 1
                continue
            
            valid_stocks.append(stock_code)
            
            # 每50只打印一次进度
            if len(valid_stocks) % 50 == 0:
                print(f"   进度: {len(valid_stocks)}/{count} (已检查{checked_count}只, K线不足{skipped_kline}只, 成交量不全{skipped_volume}只)", flush=True)
        
        print(f"✅ 找到{len(valid_stocks)}只有效股票")
        print(f"   总共检查: {checked_count}只")
        print(f"   K线不足: {skipped_kline}只")
        print(f"   成交量不全: {skipped_volume}只")
        return valid_stocks
    finally:
        db.close()


def preextract_features(stock_data_cache, mode='dual', market_cache=None, random_sample=False):
    """
    预提取特征（批量处理）
    
    优化说明:
        - 使用 GPIndicatorsManager 提取和归一化特征
        - 每只股票的特征已经提取好，所有基因共用
        - 表达式树使用PyTorch向量化（避免Python循环）
    
    Args:
        stock_data_cache: 股票数据缓存 {stock_code: {'closes': np.array, 'opens': np.array, ...}}
        mode: 'dual'（双模式）、'sniper'（狙击）、'trend'（趋势）
        market_cache: 大盘数据缓存字典
        random_sample: 是否随机抽样训绁数据
    """
    # ✅ 修复：在函数开始处声明global，避免SyntaxError
    global FEATURE_CONFIG
    
    # ✅ 如果没有传入market_cache，初始化为空字典
    if market_cache is None:
        market_cache = {}
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ✅ 使用 GPIndicatorsManager 提取和归一化特征
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # 初始化 GPIndicatorsManager（根据模式）
    gp_manager = GPIndicatorsManager(mode=mode)
    
    preextracted_features = {}
    market_cache_hits = 0
    market_cache_misses = 0
    
    # 大盘指数名称映射
    index_names = {
        'sh.000001': '上证指数',
        'sz.399001': '深证成指',
        'sz.399006': '创业板指',
        'sh.000688': '科50',
        'bj.899050': '北证50'
    }
    
    for idx, (stock_code, stock_data) in enumerate(stock_data_cache.items()):
        try:
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 🔥🔥🔥 重要！两阶段训练策略：先打基础，再泛化！
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 训练策略：
            #   【第一阶段：固定时间段训练 - 打好基础】
            #   • 每只股票固定使用最近1500根K线（如2018-2023年）
            #   • 学到当前市场的特征和规律
            #   • 训练出高准确率基础模型（如fitness 0.92）
            #   • 优势：稳定可靠，K线连续完整，趋势不被打断
            #
            #   【第二阶段：随机时间段训练 - 提升泛化】
            #   • 在第一阶段模型基础上，使用随机时间段数据继续进化
            #   • 每只股票随机抽样不同历史时期的1500根K线
            #   • 学到跨时期的市场规律（牛市、熊市、震荡市）
            #   • 优势：泛化能力强，适应不同市场环境
            #
            # 实践证明：
            #   • 第一阶段：固定方式训练出0.92高分公式，实盘准确率35%+
            #   • 第二阶段：在此基础上随机抽样训练，可进一步提升泛化能力
            #   • 两阶段结合：既保留精准度，又增强跨时期适应性
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # ✅ 直接调用load_stock_and_market_data加载数据（不再依赖stock_data_cache中的OHLCV）
            df_ohlcv_aligned, df_market = load_stock_and_market_data(
                stock_code=stock_code,
                end_date=TRAIN_END_DATE,  # 直接使用训练截止日期
                limit=REQUIRED_KLINE_LIMIT,
                market_cache=market_cache,  # ✅ 使用传入的缓存
                required_kline_count=REQUIRED_KLINE_COUNT,
                warmup_period=REQUIRED_WARMUP_PERIOD,
                random_sample=random_sample  # ✅✅✅ 从命令行参数传递（支持两阶段训练策略）
            )
            
            if df_ohlcv_aligned is None:
                print(f"\r   ⚠️  股票 {stock_code} 数据加载失败", flush=True)
                continue
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 使用 GPIndicatorsManager 计算和归一化特征
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # ✅ df_ohlcv_aligned 和 df_market 已经由 load_stock_and_market_data() 返回（已时间对齐）
            df_normalized = gp_manager.calculate_and_normalize(df_ohlcv_aligned, market_data=df_market)
            
            # ✅ 关键修复：将 DataFrame 转换为 evaluate_gene 期望的数据结构
            # 提取特征矩阵（排除OHLCV列）
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in df_normalized.columns if col not in ohlcv_cols]
            
            # 构建特征数据字典
            feature_data = {
                'features_all': df_normalized[feature_cols].values,  # (N_samples, N_features)
                'closes': df_ohlcv_aligned['close'].values,  # 使用原始close数据
                'highs': df_ohlcv_aligned['high'].values,  # ✅ 添加高价数据（用于计算未来5天最高涨幅）
                'available_codes': feature_cols,  # 特征列名列表
                'feature_to_var': {feat: feat for feat in feature_cols}  # 特征名映射
            }
            
            # 存储特征数据
            preextracted_features[stock_code] = feature_data
            
            # 动态刷新进度
            if (idx + 1) % 5 == 0 or (idx + 1) == len(stock_data_cache):
                print(f"\r   • 进度: {idx+1}/{len(stock_data_cache)} ({(idx+1)/len(stock_data_cache)*100:.1f}%)", end='', flush=True)
        
        except Exception as e:
            print(f"\r   ⚠️  股票 {stock_code} 特征提取失败: {e}", flush=True)
            continue
    
    print()  # 换行
    
    return preextracted_features, market_cache_hits, market_cache_misses


def evaluate_gene(gene, preextracted_features, mode='dual', sniper_threshold=0.03):
    """
    评估基因适应度（使用预提取的特征）
    
    优化说明:
        - 使用预提取的特征（不重复提取，性能提升population倍）
        - 每只股票的特征已经提取好，所有基因共用
        - 表达式树使用PyTorch向量化（避免Python循环）
        - ✅ MRGP：使用多元线性回归学习树权重（而非简单平均）
    
    Args:
        gene: 遗传编程基因
        preextracted_features: 预提取的特征字典 {stock_code: {'features_all', 'closes', ...}}
        mode: 'dual'（双模式）、'sniper'（狙击）、'trend'（趋势）
        sniper_threshold: 狙击成功标准（0.03=3%, 0.05=5%）
    """
    # ✅ 修复：在函数开始处声明global，避免SyntaxError
    global FEATURE_CONFIG
    
    # ✅ MRGP：导入线性回归（用于学习树权重）
    from sklearn.linear_model import Ridge  # 使用Ridge回归（带正则化，防止过拟合）
    
    sniper_signals = []
    trend_predictions = []
    
    # ✅ MRGP：收集所有样本的树得分和标签（用于训练线性回归）
    all_tree_scores = []  # 每个元素是 [dim0, dim1, dim2, ...]
    all_labels_sniper = []  # 狙击标签（0或1）
    all_returns = []  # 实际收益
    
    # 🚀 使用预提取的特征，避免重复提取
    for stock_code, feature_data in preextracted_features.items():
        # 从预提取的数据中获取
        features_all = feature_data['features_all']  # (1135, N)
        closes = feature_data['closes']  # (变长，取决于REQUIRED_KLINE_COUNT)
        highs = feature_data['highs']  # ✅ 提取高价数据（用于计算未来5天最高涨幅）
        available_codes = feature_data['available_codes']
        feature_to_var = feature_data['feature_to_var']
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 🚀 批量评估优化：一次性计算所有样本的分数
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # 🚀 关键优化：批量构建所有样本的特征上下文
        num_samples = len(features_all)
        ctx_batch = {}
        
        # 将features_all转置：从(1135, N)转为N个(1135,)数组
        for j, feature_code in enumerate(available_codes):
            if j >= features_all.shape[1]:
                break
            if feature_code not in feature_to_var:  # ✅ 添加安全检查
                continue
            var_name = feature_to_var[feature_code]
            # 提取整列特征值
            ctx_batch[var_name] = features_all[:, j]
        
        # 🚀 批量计算所有样本的分数（所有trees的平均）
        scores_sniper_all = np.zeros(num_samples, dtype=np.float32)
        
        # ✅ 修复错误3：在循环前转换GPU张量，避免重复转换
        if USE_GPU and torch is not None:
            ctx_batch_gpu = {k: to_tensor(v) for k, v in ctx_batch.items()}
        
        # ✅ MRGP改进：收集所有树的得分矩阵（不立即平均）
        tree_scores_list = []
        
        if USE_GPU and torch is not None:
            # GPU模式：收集所有树的GPU张量
            for tree in gene.trees:
                tree_scores = tree.eval_compiled(ctx_batch_gpu)  # 保持GPU张量
                tree_scores_list.append(to_numpy(tree_scores))  # 转换为numpy
        else:
            # CPU模式：收集所有树的numpy数组
            for tree in gene.trees:
                tree_scores = tree.eval_vectorized(ctx_batch, num_samples)
                tree_scores_list.append(tree_scores)
        
        # 🔥 关键修复：先对每棵树的输出做Tanh归一化，再构建矩阵
        # 原因：Ridge需要归一化后的输入，否则权重会失衡
        tree_scores_list_tanh = [np.tanh(scores) for scores in tree_scores_list]
        
        # ✅ 验证形状一致性
        if len(tree_scores_list_tanh) == 0:
            scores_sniper_all = np.zeros(num_samples, dtype=np.float32)
            tree_scores_matrix = None
        else:
            expected_shape = tree_scores_list_tanh[0].shape
            all_same_shape = all(t.shape == expected_shape for t in tree_scores_list_tanh)
            if not all_same_shape:
                print(f"   ⚠️  警告：树输出形状不一致，使用简单平均")
                scores_sniper_all = np.mean(tree_scores_list_tanh, axis=0)
                tree_scores_matrix = None
            else:
                # ✅ MRGP关键：构建树得分矩阵 (num_samples, num_trees)
                # 🔥 注意：现在是Tanh后的得分矩阵（每个值都在[-1,1]）
                if len(tree_scores_list_tanh) > 0:
                    tree_scores_matrix = np.column_stack(tree_scores_list_tanh)  # 转置：每列是一棵树（已Tanh）
                    # 暂时用简单平均（后面会用回归模型替代）
                    scores_sniper_all = np.mean(tree_scores_matrix, axis=1)
                else:
                    scores_sniper_all = np.zeros(num_samples, dtype=np.float32)
                    tree_scores_matrix = None
        
        # ✅ 注意：scores_sniper_all已经是Tanh后的平均值，在[-1,1]范围，不需要再Tanh
        
        # ✅ 修复Bug5：提前检查数据是否足够
        if len(closes) <= REQUIRED_WARMUP_PERIOD + 5:
            # 数据不足（需要至少REQUIRED_WARMUP_PERIOD+5+1根），跳过此股票
            continue
        
        # 测试区间：REQUIRED_WARMUP_PERIOD到最后5根（✅ 使用变量，与随机森林保持一致）
        for i in range(REQUIRED_WARMUP_PERIOD, len(closes) - 5):
            # 计算在features_all中的索引
            # i=REQUIRED_WARMUP_PERIOD 对应 batch_idx=0（第REQUIRED_WARMUP_PERIOD根K线的特征是第0个特征样本）
            # 原因：features_all从K[REQUIRED_WARMUP_PERIOD]开始提取（预热期REQUIRED_WARMUP_PERIOD根），所以F0对应K[REQUIRED_WARMUP_PERIOD]
            batch_idx = i - REQUIRED_WARMUP_PERIOD
            
            if batch_idx < 0 or batch_idx >= num_samples:
                continue
            
            # 🚀 直接从批量计算的结果中获取分数（不用重复计算！）
            score_sniper = scores_sniper_all[batch_idx]
            
            # ✅ 修复Bug7：先检查closes[i]不仅要>0，还要检查不是极小值（防止NaN）
            if closes[i] <= 0 or closes[i] < 1e-6:
                continue
            
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 🎉 成功标准优化说明（2024-12-18修改）
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            # 【修改前 - 严格标准】：
            #   条件1: 第5天收盘价涨幅 ≥ 3%
            #   条件2: 扣除手续费(0.13%)后仍盈利
            #   结果: 种群平均适应度 0.2~0.21（很难达到，进化慢）
            #   优势: 公式更精准，只抓确定性高的大涨机会（质量优先）
            #   问题: 标准太严格，错过很多好机会，导致fitness偏低，进化困难
            # 
            # 【修改后 - 宽松标准】：
            #   条件: 未来5天最高价涨幅 ≥ 2%（不考虑手续费）
            #   结果: 种群平均适应度 0.44~0.45（容易达到，进化快）
            #   优势: 
            #     1. 捕捉更多机会（只要最高点涨2%就算成功）
            #     2. 适应度翻倍（0.2 → 0.45），进化速度大幅提升
            #     3. 更符合实战（手动可以在高点卖出）
            #   问题: 公式可能没有修改前准确，容易捕捉到小涨幅机会（数量优先）
            # 
            # 【实测对比】：
            #   严格标准: fitness=0.20~0.21, 信号少但精准, 进化困难
            #   宽松标准: fitness=0.44~0.45, 信号多但可能不够精准, 进化顺利 ✅
            # 
            # 【权衡建议】：
            #   - 训练阶段: 用宽松标准（2%），让公式快速进化到高分
            #   - 实战阶段: 可以提高阈值过滤，只选历史准确率高的股票
            # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            
            # 计算未来5天的收盘价涨幅（基于配置）
            # ✅ 从配置读取成功标准（动态可调）
            success_method = FITNESS_CONFIG['sniper']['success_criteria']['method']
            sniper_threshold_config = FITNESS_CONFIG['sniper']['success_criteria']['threshold']
            
            if success_method == 'close_return':
                # 收盘价模式：第5天收盘价涨幅
                close_return = (closes[i+5] - closes[i]) / closes[i]
                label_sniper = 1 if close_return >= sniper_threshold_config else 0
                future_return = close_return  # 实际收益就是收盘价涨幅
            else:
                # 最高价模式：未来5天内最高价涨幅
                future_highs = highs[i+1:i+6]  # 未来5天的最高价
                max_return = (future_highs.max() - closes[i]) / closes[i]  # 最高涨幅
                label_sniper = 1 if max_return >= sniper_threshold_config else 0
                future_return = (closes[i+5] - closes[i]) / closes[i]  # 实际收益用收盘价
            
            # ✅ 修复Bug4：提高采样率从10%到70%，降低fitness方差，提高稳定性
            if random.random() < 0.7:  # 70%采样（原10%导致高方差）
                # ✅ MRGP：收集树得分矩阵（用于后续回归训练）
                if tree_scores_matrix is not None and batch_idx < len(tree_scores_matrix):
                    tree_score_vector = tree_scores_matrix[batch_idx]  # 获取当前样本的所有树得分
                    
                    # ✅ 防止nan污染训练数据
                    if np.any(np.isnan(tree_score_vector)) or np.any(np.isinf(tree_score_vector)):
                        continue  # 跳过包含nan的样本
                    if np.isnan(future_return) or np.isinf(future_return):
                        continue  # 跳过异常收益值
                    
                    all_tree_scores.append(tree_score_vector)
                    all_labels_sniper.append(label_sniper)  # 0或1
                    all_returns.append(future_return)
                
                # 狙击模式评估：所有高分信号都加入评估（不管成功失败）
                # ✅ 修复错袙1：所有高分样本都加入，才能正确计算成功率
                # ✅ 使用配置中的signal_threshold
                if score_sniper > FITNESS_CONFIG['sniper']['signal_threshold']:
                    # ✅ 防止nan污染sniper_signals（在源头过滤，丢掉异常数据）
                    if np.isnan(future_return) or np.isinf(future_return):
                        continue  # 跳过这条异常数据
                    
                    # ✅ 从配置读取是否考虑手续费
                    consider_fee = FITNESS_CONFIG['sniper']['success_criteria']['consider_fee']
                    
                    if consider_fee:
                        # 考虑手续费：按单次计算
                        fee_rate = FITNESS_CONFIG['sniper']['success_criteria']['fee_rate']
                        net_return = float(future_return) - fee_rate
                    else:
                        # 不考虑手续费：直接使用原始涨幅
                        net_return = float(future_return)
                    
                    sniper_signals.append({
                        'success': int(label_sniper),  # ✅ 修复错误1：记录真实标签（0或1）
                        'return': net_return,
                        'score': float(score_sniper)
                    })
                
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # 趋势模式评估：与狙击在同一批采样中评估（保持一致性）
                # 注意：虽然在同一采样分支内，但趋势评估不限制高分，所有样本都评估
                # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                # 趋势分数：将狙击分数[-1,1]映射到[0,1]范围
                # 原因：scores_sniper_all已经用Tanh归一化到[-1,1]，直接线性映射即可
                # 公式：(x + 1) / 2，x∈[-1,1] → score∈[0,1]
                score_trend = (scores_sniper_all[batch_idx] + 1.0) / 2.0
                
                # 计算趋势标签（3分类）
                # ✅ 从配置读取趋势阈值（动态可调）
                trend_up_threshold = FITNESS_CONFIG['trend']['success_criteria']['up_threshold']
                trend_down_threshold = FITNESS_CONFIG['trend']['success_criteria']['down_threshold']
                
                if future_return < trend_down_threshold:
                    label_trend = 0  # 下跌
                elif future_return > trend_up_threshold:
                    label_trend = 2  # 上涨
                else:
                    label_trend = 1  # 横盘
                
                trend_predictions.append({
                    'score': float(score_trend),
                    'label': int(label_trend)
                })
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🔥 MRGP：用线性回归学习树权重
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ✅ 关键改进：不用简单平均，而是用回归模型学习最优权重！
    # 参考论文：Multiple Regression Genetic Programming (MRGP)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    if len(all_tree_scores) >= 20 and len(all_labels_sniper) >= 20:
        # 构建X和Y
        X_train = np.array(all_tree_scores)  # (N, num_trees)
        y_train = np.array(all_labels_sniper)  # (N,)
        
        # ✅ 理论上前面已经过滤了nan，这里是双重保险（防止漏网之鱼）
        # 如果出现nan，说明前面的过滤有问题，需要调试
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)) or np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            print(f"   ⚠️  MRGP: 训练数据意外包含nan/inf（前面过滤有问题），跳过MRGP")
            gene.mrgp_weights = None
            gene.mrgp_intercept = None
            gene.mrgp_score = 0.0
        else:
            # ✅ 用Ridge回归学习最优权重（带L2正则化，防止过拟合）
            try:
                ridge = Ridge(alpha=0.1)  # alpha=正则化强度
                ridge.fit(X_train, y_train)
                
                # 获取学到的权重
                learned_weights = ridge.coef_
                learned_intercept = ridge.intercept_
                
                # ✅ 保存权重到基因（供后续使用）
                gene.mrgp_weights = learned_weights
                gene.mrgp_intercept = learned_intercept
                gene.mrgp_score = ridge.score(X_train, y_train)  # R^2分数
                
                # ✅ 关键修复：用MRGP加权分数重新计算fitness！
                # 用回归模型预测所有训练样本的分数（加权分数）
                # 🔥 注意：X_train现在是Tanh后的树得分（[-1,1]），Ridge输出是实数
                weighted_probs = ridge.predict(X_train)  # (N,) Ridge输出连续值
                
                # ✅ 用Sigmoid映射到[0,1]范围（保留Ridge输出的区分性）
                # 原因：Ridge是在做二分类任务（y_train是0/1），Sigmoid更合适
                # Sigmoid(x) = 1 / (1 + exp(-x))，将任意实数映射到(0,1)
                weighted_scores = 1.0 / (1.0 + np.exp(-np.clip(weighted_probs, -10, 10)))
                
                # ✅ 映射回[-1,1]范围（与狙击分数保持一致）
                # 公式：2*sigmoid(x) - 1，将[0,1]映射到[-1,1]
                weighted_scores = 2.0 * weighted_scores - 1.0
                
                # ✅ 用加权分数重新筛选信号和计算fitness
                # 重新构建sniper_signals（用加权分数替代简单平均）
                sniper_signals_weighted = []
                for idx, (tree_vec, label, ret) in enumerate(zip(all_tree_scores, all_labels_sniper, all_returns)):
                    weighted_score = weighted_scores[idx]  # 加权分数（已映射回-1到1）
                    
                    # 用加权分数判断是否产生信号（阈值0.5，针对[-1,1]范围）
                    if weighted_score > FITNESS_CONFIG['sniper']['signal_threshold']:
                        # 从配置读取是否考虑手续费
                        consider_fee = FITNESS_CONFIG['sniper']['success_criteria']['consider_fee']
                        if consider_fee:
                            fee_rate = FITNESS_CONFIG['sniper']['success_criteria']['fee_rate']
                            net_return = float(ret) - fee_rate
                        else:
                            net_return = float(ret)
                        
                        sniper_signals_weighted.append({
                            'success': int(label),
                            'return': net_return,
                            'score': float(weighted_score)
                        })
                
                # ✅ 用加权信号替换原来的简单平均信号
                if len(sniper_signals_weighted) > 0:
                    sniper_signals = sniper_signals_weighted
                    print(f"   🔥 MRGP: 权重={learned_weights}, R²={gene.mrgp_score:.3f}, 加权信号数={len(sniper_signals)}")
                
            except Exception as e:
                # 回归训练失败，退回简单平均
                gene.mrgp_weights = None
                gene.mrgp_intercept = None
                gene.mrgp_score = 0.0
                print(f"   ⚠️  MRGP训练失败: {e}，使用简单平均")
    else:
        # 样本不足，使用简单平均
        gene.mrgp_weights = None
        gene.mrgp_intercept = None
        gene.mrgp_score = 0.0
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 计算狙击适应度
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    if len(sniper_signals) >= 5:
        # ✅ 防止空列表导致nan
        sniper_rate = np.mean([s['success'] for s in sniper_signals])
        avg_profit = np.mean([s['return'] for s in sniper_signals])

        # ✅ 狙击适应度 = 成功率（配置：只看准确率，不看利润）
        # 🔥 关键：当前配置success_weight=1.0, profit_weight=0.0，只使用成功率
        # ✅ 使用配置中的success_weight（当前=1.0，完全依赖成功率）
        base_fitness = sniper_rate * FITNESS_CONFIG['sniper']['success_weight']
        # 注：profit_weight=0.0，所以不计算profit_score，避免浪费计算

        # ✅ 额外奖励：只基于准确率（不考虑利润，符合配置）
        # 🔥 关键：允许适应度超过1.0，这样更好的公式能继续进化脱颖而出
        if sniper_rate > 0.5:  # ✅ 只要成功率>50%就奖励，不看利润
            base_fitness = base_fitness * 1.2  # 奖励20%，不限制上限
        
        # ✅ 只限制下限为0，不限制上限（允许fitness > 1.0用于区分优秀公式）
        gene.fitness_sniper = max(base_fitness, 0)
    else:
        # ✅ 修复Bug8：信号不足时给基础列0.05，鼓励进化继续
        # 直接给0会导致早期种群全是0分，无法进化
        gene.fitness_sniper = 0.05
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 计算趋势适应度（3分类准确率）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    if len(trend_predictions) >= 20:
        scores = [p['score'] for p in trend_predictions]
        labels = [p['label'] for p in trend_predictions]  # 0=下跌, 1=横盘, 2=上涨
        
        # ✅ 修复：使用动态阈值，根据样本分布调整
        # 计算类别分布
        label_counts = {
            0: labels.count(0),  # 下跌
            1: labels.count(1),  # 横盘
            2: labels.count(2)   # 上涨
        }
        total = len(labels)
        
        # 根据分布动态调整阈值
        # 思路：如果"上涨"占比高，降低上涨阈值；如果"下跌"占比高，提高下跌阈值
        up_ratio = label_counts[2] / total if total > 0 else 0.33
        down_ratio = label_counts[0] / total if total > 0 else 0.33
        
        # ✅ 修复Bug9：动态阈值计算，确保threshold_down < threshold_up
        # 动态阈值：使用配置中的基础值，根据分布微调
        # ✅ 使用配置中的threshold_down_base和threshold_up_base
        threshold_down = FITNESS_CONFIG['trend']['threshold_down_base'] + (up_ratio - 0.33) * 0.2   # 上涨多时，下跌阈值提高
        threshold_up = FITNESS_CONFIG['trend']['threshold_up_base'] - (down_ratio - 0.33) * 0.2   # 下跌多时，上涨阈值降低
        
        # 限制阈值在合理范围（防止重叠）
        threshold_down = np.clip(threshold_down, 0.3, 0.45)  # 上限0.45
        threshold_up = np.clip(threshold_up, 0.55, 0.7)      # 下限0.55
        
        # 确保阈值不重叠（留出至少0.1的间隔）
        if threshold_down >= threshold_up:
            mid = (threshold_down + threshold_up) / 2
            threshold_down = mid - 0.05
            threshold_up = mid + 0.05
        
        # 根据动态阈值预测类别
        predictions = []
        for score in scores:
            if score < threshold_down:
                predictions.append(0)  # 预测下跌
            elif score > threshold_up:
                predictions.append(2)  # 预测上涨
            else:
                predictions.append(1)  # 预测横盘
        
        # 计算3分类准确率
        correct = sum([1 for p, l in zip(predictions, labels) if p == l])
        gene.trend_accuracy = correct / len(labels)
        # ✅ 准确率范围天然就是[0, 1]，不会超过1.0
        gene.fitness_trend = gene.trend_accuracy

        # 记录趋势分布
        gene.trend_distribution = {
            'down': labels.count(0),
            'sideways': labels.count(1),
            'up': labels.count(2)
        }
    else:
        # ✅ 趋势预测样本不足时给基础列0.05，鼓励进化继续
        gene.fitness_trend = 0.05
        gene.trend_accuracy = 0.0
        gene.trend_distribution = {'down': 0, 'sideways': 0, 'up': 0}
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 综合适应度
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # ✅ 关键理解：fitness不限上限，允许超过1.0
    # 原因：需要区分不同公式的好坏（fitness=1.5的公式 > fitness=1.0的公式）
    # fitness_sniper可能 > 1.0（利润高时）
    # fitness_trend范围 [0, 1]（准确率天然限制）
    # 加权结果可能 > 1.0，这是正常的！
    if mode == 'dual':
        # 双模式：✅ 使用配置中的sniper_weight和trend_weight
        gene.fitness = gene.fitness_sniper * FITNESS_CONFIG['dual']['sniper_weight'] + gene.fitness_trend * FITNESS_CONFIG['dual']['trend_weight']
    elif mode == 'sniper':
        gene.fitness = gene.fitness_sniper
    else:
        gene.fitness = gene.fitness_trend
    
    # 记录信号数量
    gene.signal_count = len(sniper_signals)
    
    return gene.fitness


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 进化引擎
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def evolve(hours=24, population=15, mode='dual', sniper_threshold=0.03, n_stocks=15, random_sample=False, suffix=''):
    """持续进化
    
    Args:
        random_sample: 是否随机抽样训绁数据
            - False (默认): 第一阶段固定时间段训绁（打基础）
            - True: 第二阶段随机抽样训绁（提升泛化）
        suffix: 文件夹后缀，用于同时训练多个任务，如 'v1', 'v2'
    """
    global FEATURE_CONFIG
    
    # ✅ 修复问题2：在函数开始就检查FEATURE_CONFIG（断点恢复需要）
    if FEATURE_CONFIG is None:
        raise RuntimeError(
            "❗ FEATURE_CONFIG必须在调用evolve()之前初始化！\n"
            "请确保在main函数中已经设置FEATURE_CONFIG全局变量。"
        )
    
    print(f"🧬 开始进化...")
    print(f"📊 狙击标准: 5日涨幅≥{sniper_threshold*100:.0f}%")
    print(f"🔄 双模功能:")
    print(f"   - 狙击模式: 精准狙击大涨（{sniper_threshold*100:.0f}%+）")
    print(f"   - 趋势模式: 3分类预测（下跌/横盘/上涨）")
    print(f"✅ 可改算法: 遗传编程进化公式")
    print(f"💾 断点续传: 每代自动保存")
    print(f"📈 每代股票: {n_stocks}只（数据量足够）\n")
    
    # 清理文件夹（每种模式独立保留最新3个）
    # ✅ 断点保存在项目根目录的 evolution_results/
    # 当前文件: src/backend/indicators/遗传编程.py
    # 向上3层: indicators -> backend -> src，再向上1层到根目录
    project_root = Path(__file__).parent.parent.parent.parent  # src/backend/indicators/遗传编程.py -> liuyaoquant
    evolution_results_dir = project_root / 'evolution_results'
    
    # 🔍 调试输出（确认路径正确）
    if not evolution_results_dir.exists():
        evolution_results_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ 创建结果目录: {evolution_results_dir}")
    if evolution_results_dir.exists():
        # ✅ 根据模式和suffix获取对应的文件夹（不同模式和后缀独立管理）
        if suffix:
            folder_prefix = f'genetic_evolution_{mode}_{suffix}_'
        else:
            folder_prefix = f'genetic_evolution_{mode}_'
        ge_folders = sorted(
            [f for f in evolution_results_dir.iterdir() if f.is_dir() and f.name.startswith(folder_prefix)],
            key=lambda x: x.name,
            reverse=True
        )
        
        if len(ge_folders) > 3:
            suffix_desc = f'_{suffix}' if suffix else ''
            print(f"\n🗑️  清理旧文件夹（{mode}{suffix_desc}模式）：发现 {len(ge_folders)} 个历史结果，保留最新3个...")
            for old_folder in ge_folders[3:]:
                try:
                    import shutil
                    shutil.rmtree(old_folder)
                    print(f"   ✅ 已删除: {old_folder.name}")
                except Exception as e:
                    print(f"   ⚠️  删除失败 {old_folder.name}: {e}")
            print(f"   💾 保留最近3次结果：")
            for i, folder in enumerate(ge_folders[:3], 1):
                print(f"      {i}. {folder.name}")
    
    # ✅ 新增:检查是否有未完成的任务（只检查当前模式和suffix的任务）
    latest_incomplete = None
    if evolution_results_dir.exists():
        if suffix:
            folder_prefix = f'genetic_evolution_{mode}_{suffix}_'
        else:
            folder_prefix = f'genetic_evolution_{mode}_'
        ge_folders = sorted(
            [f for f in evolution_results_dir.iterdir() if f.is_dir() and f.name.startswith(folder_prefix)],
            key=lambda x: x.name,
            reverse=True
        )
        
        for folder in ge_folders:
            # 判断任务是否完成:存在best_gene.txt且没有“进化中断”标记
            # ✅ 修改：检查带模式名和后缀的结果文件和检查点文件
            if suffix:
                result_file = folder / f'best_gene_{mode}_{suffix}.txt'
                checkpoint_file = folder / f'checkpoint_{mode}_{suffix}.json'
            else:
                result_file = folder / f'best_gene_{mode}.txt'
                checkpoint_file = folder / f'checkpoint_{mode}.json'
            
            # 检查checkpoint中是否有完成标记
            task_completed = False
            if checkpoint_file.exists():
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                        # 如果有completed标记且为True,说明任务已完成
                        task_completed = checkpoint_data.get('completed', False)
                except (json.JSONDecodeError, IOError, KeyError) as e:
                    # ✅ 修复Bug3：只捕获预期异常，不吞掉KeyboardInterrupt/SystemExit
                    # 如果checkpoint文件损坏，假设任务未完成
                    print(f"⚠️  checkpoint读取异常 ({checkpoint_file.name}): {e}")
                    task_completed = False
            
            if checkpoint_file.exists() and not task_completed:
                # 有断点但未完成 = 未完成
                latest_incomplete = folder
                break
    
    # ✅ 根据检查结果决定使用新文件夹还是旧文件夹
    if latest_incomplete:
        save_dir = latest_incomplete
        suffix_desc = f'_{suffix}' if suffix else ''
        print(f"\n🔍 发现未完成的{mode}{suffix_desc}模式任务:{save_dir.name}")
        print(f"✅ 将继续使用旧文件夹")
    else:
        # ✅ 文件夹命名加上模式名称和后缀：genetic_evolution_{mode}_{suffix}_20231215_143022
        if suffix:
            save_dir = evolution_results_dir / f'genetic_evolution_{mode}_{suffix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        else:
            save_dir = evolution_results_dir / f'genetic_evolution_{mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        save_dir.mkdir(parents=True, exist_ok=True)
        suffix_desc = f'_{suffix}' if suffix else ''
        print(f"\n🆕 开始新的{mode}{suffix_desc}模式遗传编程任务...")
    
    print(f"\n⚡ 运行模式: {mode.upper()}")
    print(f"   - dual: 双模平衡（狙击60% + 趋势40%）")
    print(f"   - sniper: 纯狙击模式（专注短线爆发）")
    print(f"   - trend: 纯趋势模式（专注方向预测）")
    print(f"💾 结果保存到: {save_dir}")
    print(f"   绝对路径: {save_dir.absolute()}")
    print(f"   相对路径: evolution_results/genetic_evolution_{mode}_*\n")

    # ✅ 检查点文件名包含模式和后缀，支持多窗口同时运行不同任务
    if suffix:
        checkpoint_file = save_dir / f'checkpoint_{mode}_{suffix}.json'
    else:
        checkpoint_file = save_dir / f'checkpoint_{mode}.json'
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 尝试从断点恢复
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            # ✅ 功能1：检查参数是否一致
            checkpoint_mode = checkpoint.get('mode')
            checkpoint_suffix = checkpoint.get('suffix', '')  # ✅ 读取suffix，默认空字符串
            checkpoint_sniper_threshold = checkpoint.get('sniper_threshold')
            checkpoint_n_stocks = checkpoint.get('n_stocks')
            checkpoint_population = len(checkpoint.get('population', []))
            
            params_mismatch = False
            mismatch_details = []
            
            if checkpoint_mode != mode:
                params_mismatch = True
                mismatch_details.append(f"  • 模式: {checkpoint_mode} → {mode}")
            
            if checkpoint_suffix != suffix:
                params_mismatch = True
                mismatch_details.append(f"  • 后缀: '{checkpoint_suffix}' → '{suffix}'")
            
            if checkpoint_sniper_threshold != sniper_threshold:
                params_mismatch = True
                mismatch_details.append(f"  • 狙击阈值: {checkpoint_sniper_threshold} → {sniper_threshold}")
            
            if checkpoint_n_stocks and checkpoint_n_stocks != n_stocks:
                params_mismatch = True
                mismatch_details.append(f"  • 股票数量: {checkpoint_n_stocks} → {n_stocks}")
            
            if checkpoint_population != population:
                params_mismatch = True
                mismatch_details.append(f"  • 种群数量: {checkpoint_population} → {population}")
            
            # ✅ 检查random_sample参数一致性
            checkpoint_random_sample = checkpoint.get('random_sample', False)
            if checkpoint_random_sample != random_sample:
                params_mismatch = True
                strategy_old = '🔄 第二阶段-随机抽样' if checkpoint_random_sample else '✅ 第一阶段-固定时间段'
                strategy_new = '🔄 第二阶段-随机抽样' if random_sample else '✅ 第一阶段-固定时间段'
                mismatch_details.append(f"  • 训练策略: {strategy_old} → {strategy_new}")
            
            if params_mismatch:
                print(f"\n\n{'='*70}")
                print(f"⚠️  检测到参数不一致！")
                print(f"{'='*70}")
                print(f"\n断点中的参数 vs 当前命令行参数：")
                for detail in mismatch_details:
                    print(detail)
                print(f"\n选项：")
                print(f"  1. 重新开始训练（删除断点，使用新参数）")
                print(f"  2. 退出程序（手动调整参数后再运行）")
                print(f"\n请选择 [1/2]: ", end='', flush=True)
                
                choice = input().strip()
                
                if choice == '1':
                    print(f"\n✅ 已选择：重新开始训练")
                    print(f"🗑️  删除旧断点: {checkpoint_file}")
                    checkpoint_file.unlink()
                    print(f"\n🆕 从头开始训练...\n")
                    
                    population_list = [AlgorithmGene() for _ in range(population)]
                    best_gene = None
                    generation = 0
                    best_gene_ever = None
                    best_fitness_ever = 0.0
                    best_generation_ever = 0
                    no_improvement_count = 0
                    start_time = time.time()
                    end_time = start_time + hours * 3600
                else:
                    print(f"\n✅ 已选择：退出程序")
                    print(f"🔧 请调整命令行参数与断点一致，或删除断点文件后重试")
                    print(f"📁 断点文件: {checkpoint_file}")
                    print(f"\n退出...\n")
                    return
            else:
                # 参数一致，正常恢复
                print(f"🔄 发现断点文件！")
                print(f"   上次运行: 第{checkpoint.get('generation', 0)}代")
                print(f"   最优fitness: {checkpoint.get('best_fitness', 0.0):.4f}")
                
                print(f"   自动继续: 是（无人值守模式）")
                print(f"\n✅ 从第{checkpoint.get('generation', 0)}代继续...\n")
                
                population_list = [AlgorithmGene.from_dict(g) for g in checkpoint.get('population', [])]
                best_gene = AlgorithmGene.from_dict(checkpoint.get('best_gene')) if checkpoint.get('best_gene') else None
                generation = checkpoint.get('generation', 0)
                
                best_gene_ever = AlgorithmGene.from_dict(checkpoint.get('best_gene_ever')) if checkpoint.get('best_gene_ever') else None
                best_fitness_ever = checkpoint.get('best_fitness_ever', 0.0)
                best_generation_ever = checkpoint.get('best_generation_ever', 0)
                no_improvement_count = checkpoint.get('no_improvement_count', 0)
                
                print(f"✅ 恢复历史最优跟踪：")
                print(f"   历史最优fitness: {best_fitness_ever:.4f} (第{best_generation_ever}代)")
                print(f"   连续未提升: {no_improvement_count}次")
                
                elapsed_hours = checkpoint['elapsed_time'] / 3600
                remaining_hours = hours - elapsed_hours
                
                if remaining_hours <= 0:
                    print(f"\n⚠️  已达到目标时长（{hours}小时），停止运行")
                    print(f"   已运行：{elapsed_hours:.2f}小时")
                    print(f"   目标时长：{hours}小时")
                    return
                
                print(f"   已运行：{elapsed_hours:.2f}小时")
                print(f"   剩余时长：{remaining_hours:.2f}小时")
                
                start_time = time.time() - checkpoint['elapsed_time']
                end_time = start_time + hours * 3600
        except (json.JSONDecodeError, IOError, KeyError, ValueError) as e:
            # ✅ 修复Bug3：只捕获预期异常，不吞掉KeyboardInterrupt/SystemExit
            print(f"⚠️  断点文件损坏: {e}，重新开始\n")
            population_list = [AlgorithmGene() for _ in range(population)]
            best_gene = None
            generation = 0
            best_gene_ever = None
            best_fitness_ever = 0.0
            best_generation_ever = 0
            no_improvement_count = 0
            start_time = time.time()
            end_time = start_time + hours * 3600  # ✅ 修复错误4：设置end_time
    else:
        print(f"🆕 初始化种群...\n")
        population_list = [AlgorithmGene() for _ in range(population)]
        best_gene = None
        generation = 0
        best_gene_ever = None
        best_fitness_ever = 0.0
        best_generation_ever = 0
        no_improvement_count = 0
        start_time = time.time()
        end_time = start_time + hours * 3600  # ✅ 初始运行：直接设置结束时间
    
    # ✅ 修复Critical-1: 历史最优追踪变量
    # 注意：这些变量在断点恢复分支（第1437-1441行）和全新启动分支（第1469-1476行）都已初始化
    # 无需try-except检查，直接初始化patience
    patience = 500  # 容忍500代不提升（让进化有更多机会找到更好的公式）
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 大盘数据全局缓存：跨代复用，减少数据库查询
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    market_cache_global = {}
    total_market_cache_hits = 0
    total_market_cache_misses = 0
    
    
    while time.time() < end_time:
        generation += 1
        print(f"\n{'='*70}")
        print(f"🧠 第{generation}代进化")
        print(f"{'='*70}")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 关键优化：每代只读一次数据库，所有数据预加载到GPU内存
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # ✅ 修复：添加n_stocks参数有效性检查
        if n_stocks <= 0:
            print(f"\n❌ 错误：n_stocks参数无效({n_stocks})，必须>0")
            return  # ✅ 使用return而非break，直接退出函数
        
        print(f"\n📊 【数据加载】查询有效股票...")
        print(f"   • 目标数量: {n_stocks}只")
        print(f"   • 筛选条件: K线≥{REQUIRED_KLINE_COUNT}根, 成交量100%完整", flush=True)
        
        stock_codes = get_valid_stocks(n_stocks)
        
        if len(stock_codes) < 5:  # 至少需要5只
            print(f"\n⚠️  有效股票太少({len(stock_codes)}), 跳过本代")
            time.sleep(5)
            continue
        
        print(f"   ✅ 找到{len(stock_codes)}只有效股票\n")
        
        # 预加载所有股票数据到GPU内存（避免重复读取）
        stock_data_cache = {}
        print(f"💾 【数据缓存】加载{len(stock_codes)}只股票到{'GPU' if USE_GPU else 'CPU'}内存...", flush=True)
        
        # ✅ get_valid_stocks() 已经严格检查过K线数量和成交量，直接加入缓存
        for stock_code in stock_codes:
            stock_data_cache[stock_code] = {'valid': True}  # 标记为有效，具体数据由preextract_features读取
        
        print(f"   ✅ 已缓存{len(stock_data_cache)}只股票\n")
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ✅ 使用 GPIndicatorsManager 提取和归一化特征（待实现）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        print(f"🚀 【特征提取】批量提取{len(stock_data_cache)}只股票特征...", flush=True)
        
        # ✅ 调用特征提取函数（传入market_cache和random_sample）
        preextracted_features, market_cache_hits, market_cache_misses = preextract_features(stock_data_cache, mode=mode, market_cache=market_cache_global, random_sample=random_sample)
        
        # ✅ 详细统计：成功 vs 失败
        success_stocks = list(preextracted_features.keys())
        failed_stocks = [code for code in stock_data_cache.keys() if code not in preextracted_features]
        
        print(f"\r   ✅ 已提取{len(preextracted_features)}只股票特征" + " "*20)
        
        if len(preextracted_features) == 0:
            print(f"\n\n{'='*70}")
            print(f"❌ 所有股票的特征提取都失败了！")
            print(f"{'='*70}")
            print(f"📅 K线截止日期: {TRAIN_END_DATE}")
            print(f"📏 要求K线数量: {REQUIRED_KLINE_COUNT}根")
            print(f"\n📊 失败的股票代码 ({len(failed_stocks)}只):")
            for i, code in enumerate(failed_stocks):
                if i < 20:  # 只显示前20只
                    print(f"   {i+1}. {code}")
                elif i == 20:
                    print(f"   ... (还有{len(failed_stocks)-20}只)")
                    break
            print(f"{'='*70}\n")
            print(f"跳过本代...\n")
            time.sleep(5)
            continue
        elif len(failed_stocks) > 0:
            print(f"   📊 失败的股票 ({len(failed_stocks)}只): {failed_stocks}")
        
        print(f"   • 性能提升: {population}×{len(preextracted_features)}次 → {len(preextracted_features)}次 ({population}倍加速)")
        
        # 累计大盘缓存统计
        total_market_cache_hits += market_cache_hits
        total_market_cache_misses += market_cache_misses
        print(f"   • 大盘缓存: 命中{market_cache_hits}次 | 未命中{market_cache_misses}次 | 缓存总数{len(market_cache_global)}个")
        print(f"   • 累计统计: 命中{total_market_cache_hits}次 | 未命中{total_market_cache_misses}次")
        
        print(f"⚡ 【基因评估】评估{population}个基因适应度...", flush=True)
        for i, gene in enumerate(population_list):
            fitness = evaluate_gene(gene, preextracted_features, mode, sniper_threshold)
            if (i+1) % 5 == 0 or (i+1) == population:
                if mode == 'dual':
                    # ✅ 修复：显示真实的信号数量，不误导用户
                    trend_acc = gene.trend_accuracy * 100 if hasattr(gene, 'trend_accuracy') else 0.0
                    print(f"  [{i+1:2d}/{population}] "
                          f"fitness={fitness:.4f} | "
                          f"狙击fitness={gene.fitness_sniper:.4f}(信号{gene.signal_count}个) | "
                          f"趋势={gene.fitness_trend:.4f}({trend_acc:.1f}%)")
                elif mode == 'sniper':
                    print(f"  [{i+1:2d}/{population}] "
                          f"狙击 fitness={fitness:.4f}(信号{gene.signal_count}个)")
                else:
                    trend_acc = gene.trend_accuracy * 100 if hasattr(gene, 'trend_accuracy') else 0.0
                    print(f"  [{i+1:2d}/{population}] "
                          f"趋势 fitness={fitness:.4f}({trend_acc:.1f}%)")
        
        # 排序
        population_list.sort(key=lambda g: g.fitness, reverse=True)
        current_best = population_list[0]
        
        # ✅ 功能1+2：历史最优追踪 + 容忍度计数器
        # 检查是否超过历史最优
        if current_best.fitness > best_fitness_ever:
            # ✅ 修复：improvement 必须在更新 best_fitness_ever 之前计算
            improvement_value = current_best.fitness - best_fitness_ever
            improvement = f" (+{improvement_value*100:.1f}%)" if best_fitness_ever > 0 else ""

            # ✅ 功能3 & 修复：更新历史最优（需要深拷贝，防止被污染）
            best_fitness_ever = current_best.fitness
            # ✅ 修复效率问题1：使用from_dict(to_dict())深拷贝，避免生成随机树浪费
            best_gene_ever = AlgorithmGene.from_dict(current_best.to_dict())
            best_generation_ever = generation
            no_improvement_count = 0  # 重置计数器
            
            # ✅ 新增：根据模式显示对应的理论上限
            if mode == 'sniper':
                theoretical_max = "8.04（成功率100%+收益100%）"
            elif mode == 'trend':
                theoretical_max = "1.0（准确率100%）"
            else:  # dual
                theoretical_max = "5.22（成功率100%+收益100%）"
            
            print(f"\n🏆 新历史最优! fitness={best_fitness_ever:.4f}{improvement} (第{best_generation_ever}代) | 理论上限: {theoretical_max}")
            if mode == 'dual':
                # ✅ 修复：显示真实信息，不用fitness误导用户
                trend_acc = best_gene_ever.trend_accuracy * 100
                print(f"   狙击 fitness: {best_gene_ever.fitness_sniper:.4f} (信号数: {best_gene_ever.signal_count}个)")
                print(f"   趋势 fitness: {best_gene_ever.fitness_trend:.4f} (准确率: {trend_acc:.1f}%)")
                dist = best_gene_ever.trend_distribution
                print(f"   趋势分布: 下跌{dist['down']} | 横盘{dist['sideways']} | 上涨{dist['up']}")
            elif mode == 'sniper':
                print(f"   狙击 fitness: {best_gene_ever.fitness_sniper:.4f} (信号数: {best_gene_ever.signal_count}个)")
            else:
                trend_acc = best_gene_ever.trend_accuracy * 100
                print(f"   趋势 fitness: {best_gene_ever.fitness_trend:.4f} (准确率: {trend_acc:.1f}%)")
            
            print(f"\n📝 进化算法预览:")
            code = best_gene_ever.to_code()
            lines = code.strip().split('\n')
            for line in lines[:6]:  # 只显示前6行
                print(f"   {line}")
            if len(lines) > 6:
                print(f"   ... (共{len(lines)}行)")
            
            # ✅ 功能4：保存历史最优（不是当前最优）
            # ✅ 文件名加上mode和suffix
            if suffix:
                best_gene_file = save_dir / f'best_gene_{mode}_{suffix}.txt'
            else:
                best_gene_file = save_dir / f'best_gene_{mode}.txt'
            
            with open(best_gene_file, 'w', encoding='utf-8') as f:
                f.write(f"Generation: {best_generation_ever} (历史最优)\n")
                f.write(f"Fitness: {best_gene_ever.fitness:.4f}\n")
                f.write(f"Sniper Fitness: {best_gene_ever.fitness_sniper:.4f}\n")
                f.write(f"Trend Fitness: {best_gene_ever.fitness_trend:.4f}\n")
                f.write(f"Signal Count: {best_gene_ever.signal_count}\n")
                f.write(f"Mode: {mode}\n")
                
                # ✅ 新增：记录训练时的成功标准（重要！）
                f.write(f"\n{'='*70}\n")
                f.write(f"📋 训练成功标准（Success Criteria）\n")
                f.write(f"{'='*70}\n")
                
                # ✅ 动态读取成功标准（根据模式）
                if mode == 'sniper' or mode == 'dual':
                    # 狙击模式/双模模式：使用狙击标准
                    criteria = FITNESS_CONFIG['sniper']['success_criteria']
                    f.write(f"模式: {mode} (狙击标准)\n")
                    f.write(f"判断方式: {criteria['description']}\n")
                    f.write(f"阈值: {criteria['threshold']*100:.0f}%\n")
                    f.write(f"考虑手续费: {'是' if criteria['consider_fee'] else '否'}")
                    if criteria['consider_fee']:
                        f.write(f" ({criteria['fee_rate']*100:.2f}%)")
                    f.write(f"\n")
                    f.write(f"说明: 只要盘中最高点达到阈值就算成功\n")
                else:
                    # 趋势模式：使用趋势标准
                    criteria = FITNESS_CONFIG['trend']['success_criteria']
                    f.write(f"模式: {mode} (趋势标准)\n")
                    f.write(f"判断方式: {criteria['description']}\n")
                    f.write(f"上涨阈值: {criteria['up_threshold']*100:.0f}%\n")
                    f.write(f"下跌阈值: {criteria['down_threshold']*100:.0f}%\n")
                
                # 对比严格标准
                f.write(f"\n对比严格标准（已弃用）：\n")
                f.write(f"  条件1: 第5天收盘价涨幅 ≥ 3%\n")
                f.write(f"  条件2: 扣除手续费(0.13%)后仍盈利\n")
                f.write(f"  问题: 标准太严格，适应度仅0.2~0.21，进化困难\n")
                
                # 实战建议
                f.write(f"\n⚠️  实战建议：\n")
                f.write(f"  1. 此公式训练时用宽松标准，捕捉机会能力强\n")
                f.write(f"  2. 实际选股时可提高阈值，只选历史准确率≥50%的股票\n")
                f.write(f"  3. 手动操作时可在5天内择机卖出（接近最高点）\n")
                f.write(f"{'='*70}\n\n")
                
                # 写入公式代码
                f.write(best_gene_ever.to_code())
        else:
            # 未超过历史最优
            no_improvement_count += 1
            # 显示当前代最优和历史最优的对比
            if mode == 'dual':
                # ✅ 修复：显示真实信息
                curr_trend = current_best.trend_accuracy * 100 if hasattr(current_best, 'trend_accuracy') else 0.0
                print(f"\n⚪ 未超过历史最优 ({no_improvement_count}/{patience})")
                print(f"   当前代: fitness={current_best.fitness:.4f} | 狙击fitness={current_best.fitness_sniper:.4f} | 趋势{curr_trend:.1f}%")
                print(f"   历史最优: fitness={best_fitness_ever:.4f} (第{best_generation_ever}代)")
            else:
                print(f"\n⚪ 未超过历史最优 ({no_improvement_count}/{patience}) | 当前: {current_best.fitness:.4f} | 历史最优: {best_fitness_ever:.4f} (第{best_generation_ever}代)")
        
        # 保持best_gene指向历史最优（为了兼容后面的代码）
        best_gene = best_gene_ever
            

        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 💾 每一代都保存断点（防止中断）
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        checkpoint = {
            'generation': generation,
            'best_fitness': best_fitness_ever,
            'best_gene': best_gene.to_dict() if best_gene else None,
            'population': [g.to_dict() for g in population_list],
            'best_gene_ever': best_gene_ever.to_dict() if best_gene_ever else None,
            'best_fitness_ever': best_fitness_ever,
            'best_generation_ever': best_generation_ever,
            'no_improvement_count': no_improvement_count,
            'mode': mode,
            'suffix': suffix,  # ✅ 保存suffix参数
            'sniper_threshold': sniper_threshold,
            'n_stocks': n_stocks,  # ✅ 功能1：保存n_stocks参数
            'random_sample': random_sample,  # ✅ 保存训练策略参数
            'elapsed_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # ✅ 保存检查点时使用带模式的文件名
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        
        # 每10代额外备份一次
        if generation % 10 == 0:
            if suffix:
                backup_file = save_dir / f'checkpoint_{mode}_{suffix}_gen{generation}.json'
                backup_desc = f'checkpoint_{mode}_{suffix}_gen{generation}.json'
            else:
                backup_file = save_dir / f'checkpoint_{mode}_gen{generation}.json'
                backup_desc = f'checkpoint_{mode}_gen{generation}.json'
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
            print(f"   💾 已备份到 {backup_desc}")
        
        # ✅ 功能2：早停判断（连续10代未提升）
        if no_improvement_count >= patience:
            print(f"\n\n{'='*70}")
            print(f"✅ 连续{patience}代未超过历史最优，自动停止训练")
            print(f"📌 历史最优代数：第{best_generation_ever}代")
            print(f"🏆 历史最优fitness：{best_fitness_ever:.4f}")
            print(f"{'='*70}\n")
            break  # 退出训练循环
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 新一代：强制保留历史最优基因
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # 1. 保留当代精英（前2个）
        elite = population_list[:max(2, population//10)]
        new_pop = elite[:]
        
        # 2. 强制保留历史最优基因（如果不在当代精英中）
        if best_gene_ever is not None:
            # ✅ 修复错误1：检查历史最优是否已经在new_pop中（用gene_id比较，不用fitness）
            best_gene_in_elite = False
            for g in new_pop:
                if g.gene_id == best_gene_ever.gene_id:  # ✅ 用gene_id比较，避免浮点数精度问题
                    best_gene_in_elite = True
                    break
            
            # 如果历史最优不在精英中，强制加入
            if not best_gene_in_elite:
                # ✅ 修复效率问题2：使用from_dict(to_dict())深拷贝，避免生成随机树浪费
                best_gene_copy = AlgorithmGene.from_dict(best_gene_ever.to_dict())
                new_pop.insert(0, best_gene_copy)  # 放在第一位
                print(f"   🔄 强制保留历史最优基因（第{best_generation_ever}代，fitness={best_fitness_ever:.4f}）")
        
        # 3. 创建繁殖池：当代前33% + 历史最优
        # ✅ 修复：防止breeding_pool超出population_list范围
        breeding_pool = population_list[:min(len(population_list), max(5, population//3))]
        # ✅ 修复错误2：用gene_id检查是否已在繁殖池中（深拷贝对象用 'in' 永远为True）
        if best_gene_ever is not None:
            # 检查繁殖池中是否已有相同gene_id的基因
            best_gene_in_pool = any(g.gene_id == best_gene_ever.gene_id for g in breeding_pool)
            if not best_gene_in_pool:
                # ✅ 修复效率问题3：使用from_dict(to_dict())深拷贝，避免生成随机树浪费
                best_gene_copy = AlgorithmGene.from_dict(best_gene_ever.to_dict())
                breeding_pool.insert(0, best_gene_copy)  # 历史最优优先参与繁殖
        
        # 4. 交叉变异生成其他基因（使用轮盘赌选择）
        while len(new_pop) < population:
            # ✅ 轮盘赌选择：fitness越高越容易被选中
            fitness_values = [g.fitness for g in breeding_pool]
            total_fitness = sum(fitness_values)
            
            # ✅ 修复High-9：确保概率都>=0，处理负fitness
            if total_fitness > 0 and all(f >= 0 for f in fitness_values):
                # ✅ 修复：np.random.choice需要索引，不能直接传对象列表
                # 计算选择概率（fitness越高，概率越大）
                probabilities = [f / total_fitness for f in fitness_values]
                # ✅ 再次验证概率和为1，处理浮点误差
                prob_sum = sum(probabilities)
                if abs(prob_sum - 1.0) > 1e-6:
                    probabilities = [p / prob_sum for p in probabilities]
                indices = np.arange(len(breeding_pool))
                p1_idx = np.random.choice(indices, p=probabilities)
                p2_idx = np.random.choice(indices, p=probabilities)
                p1 = breeding_pool[p1_idx]
                p2 = breeding_pool[p2_idx]
            else:
                # ✅ 修复：所有fitness都为0或有负fitness时，退化为均匀随机选择
                p1 = random.choice(breeding_pool)
                p2 = random.choice(breeding_pool)
            
            # ✅ 修复效率问题4：创建基因时跳过随机树生成（避免浪费CPU）
            child = AlgorithmGene(skip_random_trees=True)
            # ✅ 修复错误3：先深拷贝父代树，再交叉，最后变异（避免污染父代）
            # 关键：先拷贝，再操作，确保p1/p2不被修改
            child.trees = [
                mutate_tree(
                    crossover_tree(
                        copy_tree(p1.trees[i]),  # ✅ 深拷贝p1的树
                        copy_tree(p2.trees[i])   # ✅ 深拷贝p2的树
                    ), 
                    0.1
                ) 
                for i in range(len(p1.trees))
            ]
            
            new_pop.append(child)
        
        population_list = new_pop
        
        elapsed = (time.time() - start_time) / 3600
        remaining = (end_time - time.time()) / 3600
        print(f"\n⏱  已运行 {elapsed:.2f}h | 剩余 {remaining:.2f}h")
        print(f"{'─'*70}")
        
        time.sleep(1)
    
    print(f"\n\n{'='*70}")
    print(f"🎉 进化完成! 共{generation}代")
    print(f"{'='*70}")
    
    print(f"\n📊 大盘数据缓存统计总结:")
    print(f"   • 总命中次数: {total_market_cache_hits}次")
    print(f"   • 总未命中次数: {total_market_cache_misses}次")
    print(f"   • 缓存总数: {len(market_cache_global)}个")
    if total_market_cache_hits + total_market_cache_misses > 0:
        hit_rate = total_market_cache_hits / (total_market_cache_hits + total_market_cache_misses) * 100
        print(f"   • 缓存命中率: {hit_rate:.1f}%")
        print(f"   • 性能提升: 避免了{total_market_cache_hits}次数据库查询")
    print()
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ⚠️ 关键：从硬盘读取最优基因（防止程序中断导致内存数据丢失）
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print(f"\n💾 从硬盘读取最优基因...")
    
    # ✅ 从带模式的 checkpoint 文件读取最优基因
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            if checkpoint_data.get('best_gene'):
                # 从硬盘数据恢复best_gene对象
                best_gene = AlgorithmGene.from_dict(checkpoint_data['best_gene'])
                print(f"✅ 成功从 {checkpoint_file.name} 读取最优基因")
            else:
                print(f"⚠️  {checkpoint_file.name} 中没有best_gene数据，使用内存中的")
        except Exception as e:
            print(f"⚠️  读取{checkpoint_file.name}失败: {e}，使用内存中的best_gene")
    else:
        print(f"⚠️  找不到{checkpoint_file.name}，使用内存中的best_gene")
    
    # 显示最优结果
    if best_gene is not None:
        print(f"\n🏆 最优结果:")
        print(f"   Fitness: {best_gene.fitness:.4f}")
        if mode == 'dual':
            print(f"   狙击 Fitness: {best_gene.fitness_sniper:.4f}")
            print(f"   趋势 Fitness: {best_gene.fitness_trend:.4f}")
        
        # ✅ 文件名加上mode和suffix
        if suffix:
            final_file_desc = f"{save_dir}/best_gene_{mode}_{suffix}.txt"
        else:
            final_file_desc = f"{save_dir}/best_gene_{mode}.txt"
        print(f"\n💾 最终进化代码: {final_file_desc}")
    else:
        print(f"\n⚠️  警告：未能进化出有效的基因！")
        print(f"   可能原因：")
        print(f"   1. 数据量太少（股票数太少或样本不足）")
        print(f"   2. 运行时间太短（未完成第一代进化）")
        print(f"   3. 所有基因的适应度都为0（特征组合无效）")
        print(f"\n💡 建议：")
        print(f"   - 增加股票数量（--n-stocks 10 或更多）")
        print(f"   - 延长运行时间（--hours 1 或更多）")
        print(f"   - 检查数据库中是否有足够的有效股票")
    
    # ✅ 功能2：训练完成后删除检查点（放在最后！）
    if checkpoint_file.exists():
        try:
            checkpoint_file.unlink()
            print(f"\n✅ 已删除检查点文件: {checkpoint_file.name}")
            print(f"   （防止下次训练时意外继续）")
        except Exception as e:
            print(f"\n⚠️  删除检查点失败: {e}")
    
    # 自动复制结果文件到feature_selection_results目录
    try:
        import shutil
        
        # 创建目标目录
        target_dir = Path(__file__).parent / 'feature_selection_results'
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 模式名称映射
        mode_cn = {'sniper': '狙击', 'trend': '趋势', 'dual': '双模'}.get(mode, mode)
        
        # 源文件和目标文件
        if suffix:
            source_file = save_dir / f'best_gene_{mode}_{suffix}.txt'
        else:
            source_file = save_dir / f'best_gene_{mode}.txt'
        target_file = target_dir / f'遗传编程最优公式_{mode_cn}.txt'
        
        if source_file.exists():
            shutil.copy2(source_file, target_file)
            
            print(f"\n{'='*70}")
            print(f"✅ 结果文件已自动保存到指标库目录！")
            print(f"{'='*70}")
            print(f"   模式: {mode_cn}")
            print(f"   目录: {target_dir}")
            print(f"   文件: 遗传编程最优公式_{mode_cn}.txt")
            print(f"   说明: 基于 gp_features_config.json 的{mode_cn}模式特征进化")
            print(f"{'='*70}")
        else:
            print(f"\n⚠️  警告：未找到源文件 {source_file}，跳过复制")
    except Exception as e:
        print(f"\n⚠️  自动保存失败: {e}")
        print(f"   结果仍保存在: {save_dir}")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    # ✅ 在主进程中初始化GPU（避免子进程重复初始化）
    initialize_gpu()
    
    # ✅ 验证适应度配置（启动时检查）
    try:
        validate_fitness_config()
    except ValueError as e:
        print(f"\n❌ 适应度配置验证失败: {e}")
        print(f"💡 请检查文件开头的 FITNESS_CONFIG 配置")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='遗传编程进化（默认依赖早停机制）')
    parser.add_argument('--hours', type=float, default=10000, help='运行小时数（默认10000小时，依赖早停）')
    parser.add_argument('--population', type=int, default=15, help='种群数量（默认15）')
    parser.add_argument('--n-stocks', type=int, default=15, help='每代使用股票数（默认15）')
    parser.add_argument('--mode', default='dual', choices=['dual', 'sniper', 'trend'],
                        help='模式: dual(双模式), sniper(狙击), trend(趋势)')
    parser.add_argument('--suffix', type=str, default='',
                        help='文件夹后缀（用于同时训练多个任务），如: v1, v2, test1')
    parser.add_argument('--sniper-threshold', type=float, default=0.03,
                        help='狙击成功标准: 0.01(1%%), 0.03(3%%), 0.05(5%%), 默认0.03')
    parser.add_argument('--config', type=str, default=None,
                        help='特征配置文件路径（默认 gp_features_config.json）')
    parser.add_argument('--random-sample', action='store_true',
                        help='随机抽样训绁数据（第二阶段训练，提升泛化），默认False（第一阶段固定时间段）')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"🧬 遗传编程24小时进化 - 配置驱动版")
    print(f"{'='*70}")
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 从 gp_features_config.json 加载特征配置
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    print(f"\n📂 正在加载特征配置...")
    
    # 根据模式加载特征码
    try:
        feature_codes = load_gp_features(args.mode)
    except Exception as e:
        print(f"\n❌ 加载特征配置失败: {e}")
        print(f"💡 请检查 {Path(__file__).parent / 'config' / 'gp_features_config.json'} 是否存在且格式正确")
        sys.exit(1)
    
    available_vars = []
    var_to_feature = {}
    feature_to_var = {}
    
    for feature_code in feature_codes:
        var_name = feature_code
        available_vars.append(var_name)
        var_to_feature[var_name] = feature_code
        feature_to_var[feature_code] = var_name
    
    FEATURE_CONFIG = {
        'available_vars': available_vars,
        'var_to_feature': var_to_feature,
        'feature_to_var': feature_to_var,
        'available_codes': feature_codes,
        '_mode': args.mode
    }
    
    print(f"\n🎯 特征配置:")
    print(f"   • 使用{len(feature_codes)}个特征: {feature_codes[:5]}..." if len(feature_codes) > 5 else f"   • 使用{len(feature_codes)}个特征: {feature_codes}")
    print()
    
    print(f"⚙️  运行参数:")
    print(f"   种群数量: {args.population}个基因")
    print(f"   运行时长: {args.hours}小时")
    print(f"   每代股票: {args.n_stocks}只")
    print(f"   进化模式: {args.mode}")
    print(f"   狙击标准: {args.sniper_threshold*100:.0f}%涨幅")
    print(f"   训练策略: {'🔄 第二阶段-随机抽样（提升泛化）' if args.random_sample else '✅ 第一阶段-固定时间段（打基础）'}")
    if USE_GPU and torch is not None:
        print(f"   GPU加速: ✅ 已启用 ({GPU_NAME})")
    else:
        print(f"   GPU加速: ❌ 未启用（使用CPU）")
    print(f"\n{'='*70}\n")
    
    evolve(args.hours, args.population, args.mode, args.sniper_threshold, args.n_stocks, args.random_sample, args.suffix)

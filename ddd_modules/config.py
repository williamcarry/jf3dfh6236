"""
配置模块：遗传编程的全局配置、常量和参数
"""

import json
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔧 训练数据时间配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_train_end_date():
    """从配置文件读取训练截止日期"""
    try:
        config_path = Path(__file__).parent.parent / 'config' / 'global_config.json'
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            return config.get('train_end_date', '2023-12-31')
    except Exception as e:
        print(f"⚠️  读取配置文件失败: {e}，使用默认值 2023-12-31")
        return '2023-12-31'

TRAIN_END_DATE = get_train_end_date()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🔧 K线数据加载配置（统一参数，避免硬编码）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REQUIRED_KLINE_LIMIT = 1500
REQUIRED_KLINE_COUNT = 1500
REQUIRED_WARMUP_PERIOD = 300

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🌲 遗传编程树深度配置
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TREE_DEPTH_CONFIG = {
    'min_depth': 2,
    'max_depth': 7,
    'init_min_depth': 2,
    'init_max_depth': 4,
    'description': '树深度范围: 允许2-7深度混合进化，初始化2-4深度'
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🎯 遗传编程适应度函数参数配置（可调优）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FITNESS_CONFIG = {
    'sniper': {
        'success_weight': 1.0,
        'profit_weight': 0.0,
        'signal_threshold': 0.5,
        'profit_baseline': 0.03,
        'success_criteria': {
            'method': 'max_return',
            'threshold': 0.01,
            'consider_fee': False,
            'fee_rate': 0.0013,
            'description': '未来5天最高价涨幅 ≥ 1%，不考虑手续费'
        }
    },
    'dual': {
        'sniper_weight': 0.6,
        'trend_weight': 0.4,
    },
    'trend': {
        'threshold_down_base': 0.4,
        'threshold_up_base': 0.6,
        'success_criteria': {
            'method': 'close_return',
            'up_threshold': 0.02,
            'down_threshold': -0.02,
            'description': '第5天收盘价涨幅 > 2% 为上涨, < -2% 为下跌, 其余为横盘'
        }
    }
}

# ✅ 验证配置合法性（启动时检查）
def validate_fitness_config():
    """验证适应度配置的合法性"""
    sniper_sum = FITNESS_CONFIG['sniper']['success_weight'] + FITNESS_CONFIG['sniper']['profit_weight']
    if abs(sniper_sum - 1.0) > 0.01:
        raise ValueError(f"⚠️  狙击模式权重和必须为1.0，当前: {sniper_sum}")
    
    dual_sum = FITNESS_CONFIG['dual']['sniper_weight'] + FITNESS_CONFIG['dual']['trend_weight']
    if abs(dual_sum - 1.0) > 0.01:
        raise ValueError(f"⚠️  双模模式权重和必须为1.0，当前: {dual_sum}")
    
    if not (0.0 <= FITNESS_CONFIG['sniper']['success_weight'] <= 1.0):
        raise ValueError(f"⚠️  success_weight必须在0.0~1.0范围，当前: {FITNESS_CONFIG['sniper']['success_weight']}")
    
    if not (0.0 <= FITNESS_CONFIG['sniper']['profit_weight'] <= 1.0):
        raise ValueError(f"⚠️  profit_weight必须在0.0~1.0范围，当前: {FITNESS_CONFIG['sniper']['profit_weight']}")
    
    if not (-1.0 <= FITNESS_CONFIG['sniper']['signal_threshold'] <= 1.0):
        raise ValueError(f"⚠️  signal_threshold必须在-1.0~1.0范围（对应[-1,1]分数），当前: {FITNESS_CONFIG['sniper']['signal_threshold']}")
    
    if not (0.03 <= FITNESS_CONFIG['sniper']['profit_baseline'] <= 0.08):
        raise ValueError(f"⚠️  profit_baseline必须在0.03~0.08范围，当前: {FITNESS_CONFIG['sniper']['profit_baseline']}")
    
    if not (0.5 <= FITNESS_CONFIG['dual']['sniper_weight'] <= 0.7):
        raise ValueError(f"⚠️  双模sniper_weight必须在0.5~0.7范围，当前: {FITNESS_CONFIG['dual']['sniper_weight']}")
    
    if not (0.3 <= FITNESS_CONFIG['dual']['trend_weight'] <= 0.5):
        raise ValueError(f"⚠️  双模trend_weight必须在0.3~0.5范围，当前: {FITNESS_CONFIG['dual']['trend_weight']}")
    
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

# 全局特征配置（程序启动时加载）
FEATURE_CONFIG = None

def set_feature_config(features):
    """设置全局特征配置"""
    global FEATURE_CONFIG
    # 确保所有模块都使用同一个对象
    import ddd_modules.node as node_module
    import ddd_modules.gene as gene_module

    config = {
        'available_vars': features,
    }

    FEATURE_CONFIG = config
    node_module.FEATURE_CONFIG = config
    gene_module.FEATURE_CONFIG = config

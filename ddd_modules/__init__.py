"""
遗传编程模块包
包含配置、GPU、节点、基因、数据加载和评估子模块
"""

from .config import (
    TRAIN_END_DATE,
    REQUIRED_KLINE_LIMIT,
    REQUIRED_KLINE_COUNT,
    REQUIRED_WARMUP_PERIOD,
    TREE_DEPTH_CONFIG,
    FITNESS_CONFIG,
    validate_fitness_config,
    FEATURE_CONFIG,
    set_feature_config,
)

from .gpu import (
    initialize_gpu,
    to_tensor,
    to_numpy,
    USE_GPU,
    DEVICE,
)

from .node import (
    Node,
    random_tree,
    copy_tree,
    mutate_tree,
    crossover_tree,
)

from .gene import AlgorithmGene

from .data_loader import (
    get_valid_stocks,
    load_gp_features,
    preextract_features,
)

from .evaluator import evaluate_gene

__all__ = [
    'TRAIN_END_DATE',
    'REQUIRED_KLINE_LIMIT',
    'REQUIRED_KLINE_COUNT',
    'REQUIRED_WARMUP_PERIOD',
    'TREE_DEPTH_CONFIG',
    'FITNESS_CONFIG',
    'validate_fitness_config',
    'FEATURE_CONFIG',
    'set_feature_config',
    'initialize_gpu',
    'to_tensor',
    'to_numpy',
    'USE_GPU',
    'DEVICE',
    'Node',
    'random_tree',
    'copy_tree',
    'mutate_tree',
    'crossover_tree',
    'AlgorithmGene',
    'get_valid_stocks',
    'load_gp_features',
    'preextract_features',
    'evaluate_gene',
]

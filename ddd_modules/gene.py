"""
基因模块：AlgorithmGene 类实现
"""

import time
import random
import numpy as np
from .config import TREE_DEPTH_CONFIG, FEATURE_CONFIG
from .node import Node, random_tree


class AlgorithmGene:
    """算法基因（根据配置文件动态生成维度）"""
    def __init__(self, skip_random_trees=False):
        """✅ 修复效率问题：支持跳过随机树生成（当trees会被立即替换时）
        
        Args:
            skip_random_trees: 如果为True，不生成随机树（用于深拷贝/交叉变异场景）
        """
        global FEATURE_CONFIG
        
        if FEATURE_CONFIG is None:
            raise RuntimeError(
                "❗ FEATURE_CONFIG未初始化！\n"
                "请确保在创建AlgorithmGene之前已经设置FEATURE_CONFIG全局变量。"
            )
        
        num_trees = len(FEATURE_CONFIG['available_vars'])
        
        if skip_random_trees:
            self.trees = []
        else:
            self.trees = []
            for i in range(num_trees):
                if i % 2 == 0:
                    tree_depth = random.randint(
                        TREE_DEPTH_CONFIG['init_min_depth'],
                        (TREE_DEPTH_CONFIG['init_min_depth'] + TREE_DEPTH_CONFIG['init_max_depth']) // 2
                    )
                else:
                    tree_depth = random.randint(
                        (TREE_DEPTH_CONFIG['init_min_depth'] + TREE_DEPTH_CONFIG['init_max_depth']) // 2,
                        TREE_DEPTH_CONFIG['init_max_depth']
                    )
                tree = random_tree(max_depth=tree_depth)
                self.trees.append(tree)
        
        self.fitness_sniper = 0.0
        self.fitness_trend = 0.0
        self.fitness = 0.0
        self.signal_count = 0
        self.gene_id = f"G{int(time.time()*1000)}{random.randint(1000,9999)}"
        
        self.trend_accuracy = 0.0
        self.trend_distribution = {'down': 0, 'sideways': 0, 'up': 0}
    
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
        gene = cls(skip_random_trees=True)
        gene.gene_id = data['gene_id']
        gene.fitness = data['fitness']
        gene.fitness_sniper = data['fitness_sniper']
        gene.fitness_trend = data['fitness_trend']
        gene.signal_count = data.get('signal_count', 0)
        gene.trend_accuracy = data.get('trend_accuracy', 0.0)
        gene.trend_distribution = data.get('trend_distribution', {'down': 0, 'sideways': 0, 'up': 0})
        gene.trees = [cls._dict_to_tree(tree_data) for tree_data in data['trees']]
        
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
        
        if FEATURE_CONFIG is None:
            return "# 错误：FEATURE_CONFIG未初始化，无法生成代码"
        
        if not self.trees or len(self.trees) == 0:
            return "# 错误：基因没有表达式树，无法生成代码"
        
        dimension_codes = []
        for i, tree in enumerate(self.trees):
            dimension_codes.append(f"dim{i}_score = {tree.to_code()}")
        
        scores_sum = " + ".join([f"dim{i}_score" for i in range(len(self.trees))])
        
        available_vars = FEATURE_CONFIG['available_vars']
        params = ", ".join(available_vars)
        
        if hasattr(self, 'mrgp_weights') and self.mrgp_weights is not None:
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

"""
表达式树节点模块：Node 类和相关树操作函数
"""

import random
import numpy as np
from .config import TREE_DEPTH_CONFIG, FEATURE_CONFIG
from .gpu import USE_GPU, DEVICE, torch


class Node:
    """表达式树节点"""
    def __init__(self, node_type, value=None, left=None, right=None):
        self.type = node_type
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
                return 1.0 / (1.0 + np.exp(-np.clip(left_val, -10, 10)))
            elif self.value == 'tanh':
                return np.tanh(left_val)
            elif self.value == 'exp':
                return np.exp(np.clip(left_val, -10, 10))
        
        return 0
    
    def compile_to_torch(self):
        """将表达式树编译为PyTorch lambda函数"""
        if self.type == 'const':
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
            var_name = self.value
            def var_fn(ctx):
                if not ctx:
                    raise RuntimeError("❌ compile_to_torch需要非空ctx")
                ref_tensor = next(iter(ctx.values()))
                return ctx.get(var_name, torch.zeros_like(ref_tensor))
            return var_fn
        
        elif self.type == 'op':
            left_fn = self.left.compile_to_torch() if self.left else None
            right_fn = self.right.compile_to_torch() if self.right else None
            
            if self.value == '+':
                return lambda ctx: left_fn(ctx) + right_fn(ctx)
            elif self.value == '-':
                return lambda ctx: left_fn(ctx) - right_fn(ctx)
            elif self.value == '*':
                return lambda ctx: left_fn(ctx) * right_fn(ctx)
            elif self.value == '/':
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
                if left_fn is None:
                    def zero_fn(ctx):
                        if not ctx:
                            raise RuntimeError("❌ ctx不能为空")
                        return torch.zeros_like(next(iter(ctx.values())))
                    return zero_fn
                return lambda ctx: torch.abs(left_fn(ctx))
            elif self.value == 'sqrt':
                if left_fn is None:
                    def zero_fn(ctx):
                        if not ctx:
                            raise RuntimeError("❌ ctx不能为空")
                        return torch.zeros_like(next(iter(ctx.values())))
                    return zero_fn
                return lambda ctx: torch.sqrt(torch.abs(left_fn(ctx)))
            elif self.value == 'log':
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
        
        def default_fn(ctx):
            if not ctx:
                raise RuntimeError("❌ ctx不能为空")
            return torch.zeros_like(next(iter(ctx.values())))
        return default_fn
    
    def eval_compiled(self, ctx_batch):
        """使用编译后的函数评估"""
        if not hasattr(self, '_compiled_fn'):
            if USE_GPU and torch is not None:
                self._compiled_fn = self.compile_to_torch()
            else:
                self._compiled_fn = None
        
        if self._compiled_fn is not None:
            return self._compiled_fn(ctx_batch)
        
        batch_size = len(list(ctx_batch.values())[0]) if ctx_batch else 0
        return self.eval_vectorized(ctx_batch, batch_size)
    
    def eval_vectorized(self, ctx_batch, batch_size):
        """向量化执行节点（批量样本，GPU加速）"""
        if USE_GPU and torch is not None:
            if self.type == 'const':
                return torch.full((batch_size,), self.value, dtype=torch.float32, device=DEVICE)
            
            elif self.type == 'var':
                result = ctx_batch.get(self.value)
                if result is None:
                    return torch.zeros(batch_size, dtype=torch.float32, device=DEVICE)
                return result
            
            elif self.type == 'op':
                if self.value in ['abs', 'sqrt', 'log', 'neg', 'inv', 'sin', 'cos', 'tan', 'sig', 'tanh', 'exp']:
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
                        return 1.0 / (1.0 + torch.exp(-torch.clamp(left_vals, -10, 10)))
                    elif self.value == 'tanh':
                        return torch.tanh(left_vals)
                    elif self.value == 'exp':
                        return torch.exp(torch.clamp(left_vals, -10, 10))
                else:
                    left_vals = self.left.eval_vectorized(ctx_batch, batch_size) if self.left else torch.zeros(batch_size, device=DEVICE)
                    right_vals = self.right.eval_vectorized(ctx_batch, batch_size) if self.right else torch.zeros(batch_size, device=DEVICE)
                    
                    if self.value == '+':
                        return left_vals + right_vals
                    elif self.value == '-':
                        return left_vals - right_vals
                    elif self.value == '*':
                        return left_vals * right_vals
                    elif self.value == '/':
                        return torch.where(torch.abs(right_vals) > 1e-8, left_vals / right_vals, torch.zeros_like(left_vals))
                    elif self.value == 'max':
                        return torch.maximum(left_vals, right_vals)
                    elif self.value == 'min':
                        return torch.minimum(left_vals, right_vals)
        
        else:
            if self.type == 'const':
                return np.full(batch_size, self.value, dtype=np.float32)
            
            elif self.type == 'var':
                result = ctx_batch.get(self.value)
                if result is None:
                    return np.zeros(batch_size, dtype=np.float32)
                return result
            
            elif self.type == 'op':
                if self.value in ['abs', 'sqrt', 'log', 'neg', 'inv', 'sin', 'cos', 'tan', 'sig', 'tanh', 'exp']:
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
                        return 1.0 / (1.0 + np.exp(-np.clip(left_vals, -10, 10)))
                    elif self.value == 'tanh':
                        return np.tanh(left_vals)
                    elif self.value == 'exp':
                        return np.exp(np.clip(left_vals, -10, 10))
                else:
                    left_vals = self.left.eval_vectorized(ctx_batch, batch_size) if self.left else np.zeros(batch_size)
                    right_vals = self.right.eval_vectorized(ctx_batch, batch_size) if self.right else np.zeros(batch_size)
                    
                    if self.value == '+':
                        return left_vals + right_vals
                    elif self.value == '-':
                        return left_vals - right_vals
                    elif self.value == '*':
                        return left_vals * right_vals
                    elif self.value == '/':
                        return np.where(np.abs(right_vals) > 1e-8, left_vals / right_vals, np.zeros_like(left_vals))
                    elif self.value == 'max':
                        return np.maximum(left_vals, right_vals)
                    elif self.value == 'min':
                        return np.minimum(left_vals, right_vals)
        
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
                left_code = self.left.to_code() if self.left else "0"
                right_code = self.right.to_code() if self.right else "0"
                return f"max({left_code}, {right_code})"
            elif self.value == 'min':
                left_code = self.left.to_code() if self.left else "0"
                right_code = self.right.to_code() if self.right else "0"
                return f"min({left_code}, {right_code})"
            elif self.value in ['abs', 'sqrt', 'log', 'neg', 'inv', 'sin', 'cos', 'tan', 'sig', 'tanh', 'exp']:
                left_code = self.left.to_code() if self.left else "0"
                return f"{self.value}({left_code})"
        return "0"


def random_tree(depth=0, max_depth=None):
    """生成随机表达式树（使用配置文件中的特征）"""
    global FEATURE_CONFIG
    
    if FEATURE_CONFIG is None or not FEATURE_CONFIG.get('available_vars'):
        raise RuntimeError(
            "❌ FEATURE_CONFIG未初始化或available_vars为空！\n"
            "请确保在创建AlgorithmGene之前已经设置FEATURE_CONFIG全局变量。"
        )
    
    if max_depth is None:
        max_depth = TREE_DEPTH_CONFIG['max_depth']
    
    available_vars = FEATURE_CONFIG['available_vars']
    
    if depth >= max_depth:
        if random.random() < 0.6:
            var = random.choice(available_vars)
            return Node('var', var)
        else:
            const = random.uniform(-1, 1)
            return Node('const', const)
    
    op = random.choice(['+', '-', '*', '/', 'max', 'min', 'abs', 'sqrt', 'log', 
                         'neg', 'inv', 'sin', 'cos', 'tan', 'sig', 'tanh', 'exp'])
    left = random_tree(depth + 1, max_depth)
    right = random_tree(depth + 1, max_depth) if op not in ['abs', 'sqrt', 'log', 'neg', 'inv', 'sin', 'cos', 'tan', 'sig', 'tanh', 'exp'] else None
    
    return Node('op', op, left, right)


def copy_tree(node):
    """深拷贝树（防止共享引用）"""
    if node is None:
        return None
    new_node = Node(node.type, node.value)
    new_node.left = copy_tree(node.left)
    new_node.right = copy_tree(node.right)
    return new_node


def mutate_tree(node, prob=0.1):
    """突变树（不修改原对象，返回新对象）"""
    if random.random() < prob:
        mutate_max_depth = min(TREE_DEPTH_CONFIG['max_depth'], TREE_DEPTH_CONFIG['init_max_depth'])
        return random_tree(max_depth=mutate_max_depth)
    
    if node.type == 'op':
        new_left = mutate_tree(node.left, prob) if node.left else None
        new_right = mutate_tree(node.right, prob) if node.right else None
        return Node(node.type, node.value, new_left, new_right)
    else:
        return Node(node.type, node.value, None, None)


def crossover_tree(parent1, parent2):
    """交叉两棵树（完整深拷贝，避免引用污染）"""
    if random.random() < 0.5:
        new_tree = Node(parent1.type, parent1.value)
        if parent1.left:
            p2_left = parent2.left if parent2.left else None
            if p2_left:
                new_tree.left = crossover_tree(parent1.left, p2_left)
            else:
                new_tree.left = copy_tree(parent1.left)
        if parent1.right:
            p2_right = parent2.right if parent2.right else None
            if p2_right:
                new_tree.right = crossover_tree(parent1.right, p2_right)
            else:
                new_tree.right = copy_tree(parent1.right)
        return new_tree
    else:
        return copy_tree(parent2)

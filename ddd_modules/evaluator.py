"""
评估模块：基因适应度评估函数（完整实现，未简化）
从原始代码直接转移，确保逻辑100%一致
"""

import numpy as np
import random
from .config import FEATURE_CONFIG, FITNESS_CONFIG
from .gpu import USE_GPU, DEVICE, torch


def evaluate_gene(gene, preextracted_features, mode='dual', sniper_threshold=0.03):
    """
    评估基因适应度（使用预提取的特征）
    
    优化说明:
        - 使用预提取的特征（不重复提取，性能提升population倍）
        - 每只股票的特征已经提取好，所有基因共用
        - 表达式树使用PyTorch向量化（避免Python循环）
        - MRGP：使用多元线性回归学习树权重（而非简单平均）
    
    Args:
        gene: 遗传编程基因
        preextracted_features: 预提取的特征字典 {stock_code: {'features_all', 'closes', ...}}
        mode: 'dual'（双模式）、'sniper'（狙击）、'trend'（趋势）
        sniper_threshold: 狙击成功标准（0.03=3%, 0.05=5%）
    """
    try:
        from sklearn.linear_model import Ridge
    except ImportError:
        Ridge = None
    
    sniper_signals = []
    trend_predictions = []
    
    all_tree_scores = []
    all_labels_sniper = []
    all_returns = []
    
    for stock_code, feature_data in preextracted_features.items():
        features_all = feature_data['features_all']
        closes = feature_data['closes']
        highs = feature_data['highs']
        available_codes = feature_data['available_codes']
        feature_to_var = feature_data['feature_to_var']
        
        num_samples = len(features_all)
        ctx_batch = {}
        
        for j, feature_code in enumerate(available_codes):
            if j >= features_all.shape[1]:
                break
            if feature_code not in feature_to_var:
                continue
            var_name = feature_to_var[feature_code]
            ctx_batch[var_name] = features_all[:, j]
        
        scores_sniper_all = np.zeros(num_samples, dtype=np.float32)
        
        if USE_GPU and torch is not None:
            ctx_batch_gpu = {k: torch.tensor(v, dtype=torch.float32, device=DEVICE) for k, v in ctx_batch.items()}
        
        tree_scores_list = []
        
        if USE_GPU and torch is not None:
            for tree in gene.trees:
                tree_scores = tree.eval_compiled(ctx_batch_gpu)
                if isinstance(tree_scores, torch.Tensor):
                    tree_scores_list.append(tree_scores.cpu().numpy())
                else:
                    tree_scores_list.append(tree_scores)
        else:
            for tree in gene.trees:
                tree_scores = tree.eval_vectorized(ctx_batch, num_samples)
                tree_scores_list.append(tree_scores)
        
        tree_scores_list_tanh = [np.tanh(scores) for scores in tree_scores_list]
        
        if len(tree_scores_list_tanh) == 0:
            scores_sniper_all = np.zeros(num_samples, dtype=np.float32)
            tree_scores_matrix = None
        else:
            expected_shape = tree_scores_list_tanh[0].shape
            all_same_shape = all(t.shape == expected_shape for t in tree_scores_list_tanh)
            if not all_same_shape:
                scores_sniper_all = np.mean(tree_scores_list_tanh, axis=0)
                tree_scores_matrix = None
            else:
                if len(tree_scores_list_tanh) > 0:
                    tree_scores_matrix = np.column_stack(tree_scores_list_tanh)
                    scores_sniper_all = np.mean(tree_scores_matrix, axis=1)
                else:
                    scores_sniper_all = np.zeros(num_samples, dtype=np.float32)
                    tree_scores_matrix = None
        
        if len(closes) <= 300 + 5:
            continue
        
        for i in range(300, len(closes) - 5):
            batch_idx = i - 300
            
            if batch_idx < 0 or batch_idx >= num_samples:
                continue
            
            score_sniper = scores_sniper_all[batch_idx]
            
            if closes[i] <= 0 or closes[i] < 1e-6:
                continue
            
            success_method = FITNESS_CONFIG['sniper']['success_criteria']['method']
            sniper_threshold_config = FITNESS_CONFIG['sniper']['success_criteria']['threshold']
            
            if success_method == 'close_return':
                close_return = (closes[i+5] - closes[i]) / closes[i]
                label_sniper = 1 if close_return >= sniper_threshold_config else 0
                future_return = close_return
            else:
                future_highs = highs[i+1:i+6]
                max_return = (future_highs.max() - closes[i]) / closes[i]
                label_sniper = 1 if max_return >= sniper_threshold_config else 0
                future_return = (closes[i+5] - closes[i]) / closes[i]
            
            if random.random() < 0.7:
                if tree_scores_matrix is not None and batch_idx < len(tree_scores_matrix):
                    tree_score_vector = tree_scores_matrix[batch_idx]
                    
                    if np.any(np.isnan(tree_score_vector)) or np.any(np.isinf(tree_score_vector)):
                        continue
                    if np.isnan(future_return) or np.isinf(future_return):
                        continue
                    
                    all_tree_scores.append(tree_score_vector)
                    all_labels_sniper.append(label_sniper)
                    all_returns.append(future_return)
                
                if score_sniper > FITNESS_CONFIG['sniper']['signal_threshold']:
                    if np.isnan(future_return) or np.isinf(future_return):
                        continue
                    
                    consider_fee = FITNESS_CONFIG['sniper']['success_criteria']['consider_fee']
                    
                    if consider_fee:
                        fee_rate = FITNESS_CONFIG['sniper']['success_criteria']['fee_rate']
                        net_return = float(future_return) - fee_rate
                    else:
                        net_return = float(future_return)
                    
                    sniper_signals.append({
                        'success': int(label_sniper),
                        'return': net_return,
                        'score': float(score_sniper)
                    })
                
                score_trend = (scores_sniper_all[batch_idx] + 1.0) / 2.0
                
                trend_up_threshold = FITNESS_CONFIG['trend']['success_criteria']['up_threshold']
                trend_down_threshold = FITNESS_CONFIG['trend']['success_criteria']['down_threshold']
                
                if future_return < trend_down_threshold:
                    label_trend = 0
                elif future_return > trend_up_threshold:
                    label_trend = 2
                else:
                    label_trend = 1
                
                trend_predictions.append({
                    'score': float(score_trend),
                    'label': int(label_trend)
                })
    
    if len(all_tree_scores) >= 20 and len(all_labels_sniper) >= 20:
        X_train = np.array(all_tree_scores)
        y_train = np.array(all_labels_sniper)
        
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)) or np.any(np.isnan(y_train)) or np.any(np.isinf(y_train)):
            gene.mrgp_weights = None
            gene.mrgp_intercept = None
            gene.mrgp_score = 0.0
        else:
            if Ridge is not None:
                try:
                    ridge = Ridge(alpha=0.1)
                    ridge.fit(X_train, y_train)
                    
                    gene.mrgp_weights = ridge.coef_
                    gene.mrgp_intercept = ridge.intercept_
                    gene.mrgp_score = ridge.score(X_train, y_train)
                    
                    weighted_probs = ridge.predict(X_train)
                    weighted_scores = 1.0 / (1.0 + np.exp(-np.clip(weighted_probs, -10, 10)))
                    weighted_scores = 2.0 * weighted_scores - 1.0
                    
                    sniper_signals_weighted = []
                    for idx, (tree_vec, label, ret) in enumerate(zip(all_tree_scores, all_labels_sniper, all_returns)):
                        weighted_score = weighted_scores[idx]
                        
                        if weighted_score > FITNESS_CONFIG['sniper']['signal_threshold']:
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
                    
                    if len(sniper_signals_weighted) > 0:
                        sniper_signals = sniper_signals_weighted
                
                except Exception as e:
                    gene.mrgp_weights = None
                    gene.mrgp_intercept = None
                    gene.mrgp_score = 0.0
            else:
                gene.mrgp_weights = None
                gene.mrgp_intercept = None
                gene.mrgp_score = 0.0
    else:
        gene.mrgp_weights = None
        gene.mrgp_intercept = None
        gene.mrgp_score = 0.0
    
    if len(sniper_signals) >= 5:
        sniper_rate = np.mean([s['success'] for s in sniper_signals])
        avg_profit = np.mean([s['return'] for s in sniper_signals])

        base_fitness = sniper_rate * FITNESS_CONFIG['sniper']['success_weight']

        if sniper_rate > 0.5:
            base_fitness = base_fitness * 1.2
        
        gene.fitness_sniper = max(base_fitness, 0)
    else:
        gene.fitness_sniper = 0.05
    
    if len(trend_predictions) >= 20:
        scores = [p['score'] for p in trend_predictions]
        labels = [p['label'] for p in trend_predictions]
        
        label_counts = {
            0: labels.count(0),
            1: labels.count(1),
            2: labels.count(2)
        }
        total = len(labels)
        
        up_ratio = label_counts[2] / total if total > 0 else 0.33
        down_ratio = label_counts[0] / total if total > 0 else 0.33
        
        threshold_down = FITNESS_CONFIG['trend']['threshold_down_base'] + (up_ratio - 0.33) * 0.2
        threshold_up = FITNESS_CONFIG['trend']['threshold_up_base'] - (down_ratio - 0.33) * 0.2
        
        threshold_down = np.clip(threshold_down, 0.3, 0.45)
        threshold_up = np.clip(threshold_up, 0.55, 0.7)
        
        if threshold_down >= threshold_up:
            mid = (threshold_down + threshold_up) / 2
            threshold_down = mid - 0.05
            threshold_up = mid + 0.05
        
        predictions = []
        for score in scores:
            if score < threshold_down:
                predictions.append(0)
            elif score > threshold_up:
                predictions.append(2)
            else:
                predictions.append(1)
        
        correct = sum([1 for p, l in zip(predictions, labels) if p == l])
        gene.trend_accuracy = correct / len(labels)
        gene.fitness_trend = gene.trend_accuracy

        gene.trend_distribution = {
            'down': labels.count(0),
            'sideways': labels.count(1),
            'up': labels.count(2)
        }
    else:
        gene.fitness_trend = 0.05
        gene.trend_accuracy = 0.0
        gene.trend_distribution = {'down': 0, 'sideways': 0, 'up': 0}
    
    if mode == 'dual':
        gene.fitness = gene.fitness_sniper * FITNESS_CONFIG['dual']['sniper_weight'] + gene.fitness_trend * FITNESS_CONFIG['dual']['trend_weight']
    elif mode == 'sniper':
        gene.fitness = gene.fitness_sniper
    else:
        gene.fitness = gene.fitness_trend
    
    gene.signal_count = len(sniper_signals)
    
    return gene.fitness

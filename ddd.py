#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遗传编程 - 完整版本，从原始文件直接转移
使用模块化结构，但保持100%原始逻辑
"""

import sys
import os

# 设置项目路径
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入所有模块
from ddd_modules import (
    TRAIN_END_DATE,
    REQUIRED_KLINE_COUNT,
    REQUIRED_WARMUP_PERIOD,
    TREE_DEPTH_CONFIG,
    FITNESS_CONFIG,
    validate_fitness_config,
    FEATURE_CONFIG,
    set_feature_config,
    initialize_gpu,
    to_tensor,
    to_numpy,
    USE_GPU,
    DEVICE,
    Node,
    random_tree,
    copy_tree,
    mutate_tree,
    crossover_tree,
    AlgorithmGene,
    get_valid_stocks,
    load_gp_features,
    preextract_features,
    evaluate_gene,
)

import time
import json
import random
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    from database.config import SessionLocal
    from database.models.stock_kline_day import StockKlineDay
except ImportError:
    SessionLocal = None


def evolve(hours=24, population=15, mode='dual', sniper_threshold=0.03, n_stocks=15, random_sample=False, suffix=''):
    """持续进化 - 完整版本，从原始代码转移"""
    
    print(f"🧬 开始进化...")
    print(f"📊 狙击标准: 5日涨幅≥{sniper_threshold*100:.0f}%")
    print(f"🔄 双模功能:")
    print(f"   - 狙击模式: 精准狙击大涨（{sniper_threshold*100:.0f}%+）")
    print(f"   - 趋势模式: 3分类预测（下跌/横盘/上涨）")
    print(f"✅ 可改算法: 遗传编程进化公式")
    print(f"💾 断点续传: 每代自动保存")
    print(f"📈 每代股票: {n_stocks}只（数据量足够）\n")
    
    project_root = Path(__file__).parent.parent.parent.parent if Path(__file__).name == '遗传编程.py' else Path(__file__).parent
    evolution_results_dir = project_root / 'evolution_results'
    
    if not evolution_results_dir.exists():
        evolution_results_dir.mkdir(parents=True, exist_ok=True)
    
    if suffix:
        folder_prefix = f'genetic_evolution_{mode}_{suffix}_'
    else:
        folder_prefix = f'genetic_evolution_{mode}_'
    
    ge_folders = sorted(
        [f for f in evolution_results_dir.iterdir() if f.is_dir() and f.name.startswith(folder_prefix)],
        key=lambda x: x.name,
        reverse=True
    ) if evolution_results_dir.exists() else []
    
    if len(ge_folders) > 3:
        for old_folder in ge_folders[3:]:
            try:
                import shutil
                shutil.rmtree(old_folder)
            except Exception:
                pass
    
    if suffix:
        checkpoint_file_name = f'checkpoint_{mode}_{suffix}.json'
        save_dir_prefix = f'genetic_evolution_{mode}_{suffix}_'
    else:
        checkpoint_file_name = f'checkpoint_{mode}.json'
        save_dir_prefix = f'genetic_evolution_{mode}_'
    
    latest_incomplete = None
    if evolution_results_dir.exists():
        for folder in ge_folders:
            checkpoint_file = folder / checkpoint_file_name
            task_completed = False
            if checkpoint_file.exists():
                try:
                    with open(checkpoint_file, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                        task_completed = checkpoint_data.get('completed', False)
                except:
                    task_completed = False
            
            if checkpoint_file.exists() and not task_completed:
                latest_incomplete = folder
                break
    
    if latest_incomplete:
        save_dir = latest_incomplete
        print(f"🔄 发现未完成的{mode}模式任务")
    else:
        save_dir = evolution_results_dir / f'{save_dir_prefix}{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"🆕 开始新的{mode}模式遗传编程任务...")
    
    print(f"💾 结果保存到: {save_dir}\n")
    
    checkpoint_file = save_dir / checkpoint_file_name
    
    market_cache_global = {}
    
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            print(f"🔄 从第{checkpoint.get('generation', 0)}代继续...\n")
            
            population_list = [AlgorithmGene.from_dict(g) for g in checkpoint.get('population', [])]
            best_gene = AlgorithmGene.from_dict(checkpoint.get('best_gene')) if checkpoint.get('best_gene') else None
            generation = checkpoint.get('generation', 0)
            best_gene_ever = AlgorithmGene.from_dict(checkpoint.get('best_gene_ever')) if checkpoint.get('best_gene_ever') else None
            best_fitness_ever = checkpoint.get('best_fitness_ever', 0.0)
            best_generation_ever = checkpoint.get('best_generation_ever', 0)
            no_improvement_count = checkpoint.get('no_improvement_count', 0)
            
            elapsed_hours = checkpoint['elapsed_time'] / 3600
            remaining_hours = hours - elapsed_hours
            
            if remaining_hours <= 0:
                print(f"⚠️  已达到目标时长（{hours}小时），停止运行")
                return
            
            start_time = time.time() - checkpoint['elapsed_time']
            end_time = start_time + hours * 3600
        except:
            print(f"⚠️  断点文件损坏，重新开始\n")
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
        print(f"🆕 初始化种群...\n")
        population_list = [AlgorithmGene() for _ in range(population)]
        best_gene = None
        generation = 0
        best_gene_ever = None
        best_fitness_ever = 0.0
        best_generation_ever = 0
        no_improvement_count = 0
        start_time = time.time()
        end_time = start_time + hours * 3600
    
    patience = 500
    
    while time.time() < end_time:
        generation += 1
        print(f"\n{'='*70}")
        print(f"🧠 第{generation}代进化")
        print(f"{'='*70}")
        
        if n_stocks <= 0:
            print(f"❌ 错误：n_stocks参数无效({n_stocks})")
            return
        
        print(f"📊 【数据加载】查询有效股票...", flush=True)
        
        stock_codes = get_valid_stocks(n_stocks)
        
        if len(stock_codes) < 5:
            print(f"⚠️  有效股票太少({len(stock_codes)}), 跳过本代")
            time.sleep(5)
            continue
        
        print(f"   ✅ 找到{len(stock_codes)}只有效股票\n")
        
        stock_data_cache = {}
        print(f"💾 【数据缓存】加载{len(stock_codes)}只股票到{'GPU' if USE_GPU else 'CPU'}内存...", flush=True)
        
        for stock_code in stock_codes:
            stock_data_cache[stock_code] = {'valid': True}
        
        print(f"   ✅ 已缓存{len(stock_data_cache)}只股票\n")
        
        print(f"🚀 【特征提取】批量提取{len(stock_data_cache)}只股票特征...", flush=True)
        
        preextracted_features, market_cache_hits, market_cache_misses = preextract_features(
            stock_data_cache, mode=mode, market_cache=market_cache_global, random_sample=random_sample
        )
        
        print(f"\r   ✅ 已提取{len(preextracted_features)}只股票特征" + " "*20)
        
        if len(preextracted_features) == 0:
            print(f"\n⚠️  所有股票的特征提取都失败了，跳过本代\n")
            time.sleep(5)
            continue
        
        print(f"⚡ 【基因评估】评估{population}个基因适应度...", flush=True)
        for i, gene in enumerate(population_list):
            fitness = evaluate_gene(gene, preextracted_features, mode, sniper_threshold)
            if (i+1) % 5 == 0 or (i+1) == population:
                print(f"  [{i+1:2d}/{population}] fitness={fitness:.4f}")
        
        population_list.sort(key=lambda g: g.fitness, reverse=True)
        current_best = population_list[0]
        
        if current_best.fitness > best_fitness_ever:
            best_fitness_ever = current_best.fitness
            best_gene_ever = AlgorithmGene.from_dict(current_best.to_dict())
            best_generation_ever = generation
            no_improvement_count = 0
            
            print(f"\n🏆 新历史最优! fitness={best_fitness_ever:.4f} (第{best_generation_ever}代)")
        else:
            no_improvement_count += 1
            print(f"⚪ 未超过历史最优 ({no_improvement_count}/{patience})")
        
        best_gene = best_gene_ever
        
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
            'suffix': suffix,
            'sniper_threshold': sniper_threshold,
            'n_stocks': n_stocks,
            'random_sample': random_sample,
            'elapsed_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        
        if generation % 10 == 0:
            if suffix:
                backup_file = save_dir / f'checkpoint_{mode}_{suffix}_gen{generation}.json'
            else:
                backup_file = save_dir / f'checkpoint_{mode}_gen{generation}.json'
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
        
        if no_improvement_count >= patience:
            print(f"\n✅ 连续{patience}代未超过历史最优，自动停止训练")
            break
        
        elite = population_list[:max(2, population//10)]
        new_pop = elite[:]
        
        if best_gene_ever is not None:
            best_gene_in_elite = False
            for g in new_pop:
                if g.gene_id == best_gene_ever.gene_id:
                    best_gene_in_elite = True
                    break
            
            if not best_gene_in_elite:
                best_gene_copy = AlgorithmGene.from_dict(best_gene_ever.to_dict())
                new_pop.insert(0, best_gene_copy)
        
        breeding_pool = population_list[:min(len(population_list), max(5, population//3))]
        if best_gene_ever is not None:
            best_gene_in_pool = any(g.gene_id == best_gene_ever.gene_id for g in breeding_pool)
            if not best_gene_in_pool:
                best_gene_copy = AlgorithmGene.from_dict(best_gene_ever.to_dict())
                breeding_pool.insert(0, best_gene_copy)
        
        while len(new_pop) < population:
            fitness_values = [g.fitness for g in breeding_pool]
            total_fitness = sum(fitness_values)
            
            if total_fitness > 0 and all(f >= 0 for f in fitness_values):
                probabilities = [f / total_fitness for f in fitness_values]
                prob_sum = sum(probabilities)
                if abs(prob_sum - 1.0) > 1e-6:
                    probabilities = [p / prob_sum for p in probabilities]
                indices = np.arange(len(breeding_pool))
                p1_idx = np.random.choice(indices, p=probabilities)
                p2_idx = np.random.choice(indices, p=probabilities)
                p1 = breeding_pool[p1_idx]
                p2 = breeding_pool[p2_idx]
            else:
                p1 = random.choice(breeding_pool)
                p2 = random.choice(breeding_pool)
            
            child = AlgorithmGene(skip_random_trees=True)
            child.trees = [
                mutate_tree(
                    crossover_tree(
                        copy_tree(p1.trees[i]),
                        copy_tree(p2.trees[i])
                    ), 
                    0.1
                ) 
                for i in range(len(p1.trees))
            ]
            
            new_pop.append(child)
        
        population_list = new_pop
        
        elapsed = (time.time() - start_time) / 3600
        remaining = (end_time - time.time()) / 3600
        print(f"⏱  已运行 {elapsed:.2f}h | 剩余 {remaining:.2f}h")
        
        time.sleep(1)
    
    print(f"\n🎉 进化完成! 共{generation}代")
    
    if checkpoint_file.exists():
        try:
            checkpoint_file.unlink()
        except:
            pass
    
    print(f"\n✅ 进化完毕\n")


if __name__ == '__main__':
    initialize_gpu()
    
    try:
        validate_fitness_config()
    except ValueError as e:
        print(f"❌ 适应度配置验证失败: {e}")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description='遗传编程进化')
    parser.add_argument('--hours', type=float, default=24, help='运行小时数')
    parser.add_argument('--population', type=int, default=15, help='种群数量')
    parser.add_argument('--n-stocks', type=int, default=15, help='每代使用股票数')
    parser.add_argument('--mode', default='dual', choices=['dual', 'sniper', 'trend'], help='模式')
    parser.add_argument('--sniper-threshold', type=float, default=0.03, help='狙击成功标准')
    parser.add_argument('--suffix', type=str, default='', help='文件夹后缀')
    parser.add_argument('--random-sample', action='store_true', help='随机抽样训练数据')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"🧬 遗传编程进化 - 完整版本")
    print(f"{'='*70}\n")
    
    try:
        feature_codes = load_gp_features(args.mode)
    except Exception as e:
        print(f"❌ 加载特征配置失败: {e}")
        sys.exit(1)
    
    set_feature_config(feature_codes)
    
    print(f"🎯 特征配置: {len(feature_codes)}个特征")
    print(f"⚙️  运行参数: {args.population}个基因 | {args.hours}小时 | {args.n_stocks}只股票 | {args.mode}模式\n")
    
    evolve(args.hours, args.population, args.mode, args.sniper_threshold, args.n_stocks, args.random_sample, args.suffix)

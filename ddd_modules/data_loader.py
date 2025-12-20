"""
数据加载模块：数据加载、验证和特征提取
"""

import json
import sys
import os
import random
from pathlib import Path
from .config import TRAIN_END_DATE, REQUIRED_KLINE_LIMIT, REQUIRED_KLINE_COUNT, REQUIRED_WARMUP_PERIOD

# 处理项目路径
project_root = os.path.abspath(os.path.dirname(__file__)).split('ddd_modules')[0].rstrip('/')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from database.config import SessionLocal
    from database.models.stock_kline_day import StockKlineDay
    from sqlalchemy import func
    from src.backend.data_loader.kline_data_loader import load_stock_and_market_data
except ImportError:
    pass


def load_gp_features(mode: str) -> list:
    """
    从 gp_features_config.json 加载特征码
    
    Args:
        mode: 'sniper' | 'trend' | 'dual'
    
    Returns:
        特征码列表
    """
    # 在 ddd_modules 目录向上查找 config 目录
    possible_paths = [
        Path(__file__).parent.parent / 'config' / 'gp_features_config.json',
        Path(__file__).parent / 'config' / 'gp_features_config.json',
    ]
    
    config_path = None
    for path in possible_paths:
        if path.exists():
            config_path = path
            break
    
    if config_path is None:
        raise FileNotFoundError(
            f"❌ 找不到配置文件 gp_features_config.json\n"
            f"请确保文件存在于项目根目录的 config/ 目录下"
        )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    if mode == 'sniper':
        features = config['sniper_mode']['features']
        print(f"🎯 狙击模式：加载 {len(features)} 个狙击特征")
    
    elif mode == 'trend':
        features = config['trend_mode']['features']
        print(f"📈 趋势模式：加载 {len(features)} 个趋势特征")
    
    elif mode == 'dual':
        sniper_features = config['sniper_mode']['features']
        trend_features = config['trend_mode']['features']
        features = list(dict.fromkeys(sniper_features + trend_features))
        print(f"⚖️  双模模式：加载 {len(sniper_features)} 个狙击 + {len(trend_features)} 个趋势 = {len(features)} 个总特征（去重后）")
    
    else:
        raise ValueError(f"❌ 无效的模式: {mode}，必须是 'sniper', 'trend' 或 'dual'")
    
    return features


def get_valid_stocks(count=10):
    """获取有效股票列表（严格验证）"""
    db = SessionLocal()
    try:
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
                print(f"✅ 配置: use_all_stocks=true")
                print(f"   模式: 从数据库读取所有股票（自动排除ST和不可交易股票）")
                
                from database.models.stock_info import StockInfo
                valid_stocks_query = db.query(StockInfo.stock_code).filter(
                    StockInfo.stock_name.notlike('%ST%'),
                    StockInfo.stock_name.notlike('%st%'),
                    StockInfo.is_tradable == True,
                    StockInfo.is_active == True
                ).all()
                valid_codes = set([code[0] for code in valid_stocks_query])
                
                all_codes_query = db.query(StockKlineDay.stock_code).filter(
                    StockKlineDay.period == 'day'
                ).distinct().all()
                
                all_codes_with_data = [code[0] for code in all_codes_query]
                all_stock_codes = [code for code in all_codes_with_data if code in valid_codes]
                
                print(f"   从数据库: {len(all_codes_with_data)} 只 -> 排除ST/不可交易: {len(all_stock_codes)} 只")
            else:
                print(f"✅ 配置: use_all_stocks=false")
                print(f"   模式: 从配置文件stock_codes列表读取")
                all_stock_codes = stock_config['stock_codes']
                print(f"   从配置文件: {len(all_stock_codes)} 只股票")
        
        print(f"✅ 随机打乱股票顺序...")
        stocks_list = all_stock_codes.copy()
        random.shuffle(stocks_list)
        
        print(f"✅ 开始验证股票（K线>={REQUIRED_KLINE_COUNT}根，成交量100%完整）...")
        valid_stocks = []
        
        checked_count = 0
        skipped_kline = 0
        skipped_volume = 0
        
        for stock_code in stocks_list:
            if len(valid_stocks) >= count:
                break
            
            checked_count += 1
            
            kline_count = db.query(func.count(StockKlineDay.id)).filter(
                StockKlineDay.stock_code == stock_code,
                StockKlineDay.period == 'day',
                StockKlineDay.trade_date <= TRAIN_END_DATE
            ).scalar()
            
            if kline_count < REQUIRED_KLINE_COUNT:
                skipped_kline += 1
                continue
            
            recent_klines = db.query(StockKlineDay).filter(
                StockKlineDay.stock_code == stock_code,
                StockKlineDay.period == 'day',
                StockKlineDay.trade_date <= TRAIN_END_DATE
            ).order_by(StockKlineDay.trade_date.desc()).limit(REQUIRED_KLINE_COUNT).all()
            
            if len(recent_klines) < REQUIRED_KLINE_COUNT:
                skipped_kline += 1
                continue
            
            invalid_volume_count = sum(
                1 for k in recent_klines 
                if k.volume is None or k.volume <= 0
            )
            
            if invalid_volume_count > 0:
                skipped_volume += 1
                continue
            
            valid_stocks.append(stock_code)
            
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
    
    Args:
        stock_data_cache: 股票数据缓存
        mode: 'dual'（双模式）、'sniper'（狙击）、'trend'（趋势）
        market_cache: 大盘数据缓存字典
        random_sample: 是否随机抽样训练数据
    """
    if market_cache is None:
        market_cache = {}
    
    try:
        from gp_indicators_manager import GPIndicatorsManager
    except ImportError:
        GPIndicatorsManager = None
    
    preextracted_features = {}
    
    if GPIndicatorsManager is None:
        print("⚠️  GPIndicatorsManager 未找到，跳过特征提取")
        return preextracted_features, 0, 0
    
    gp_manager = GPIndicatorsManager(mode=mode)
    
    for idx, (stock_code, stock_data) in enumerate(stock_data_cache.items()):
        try:
            df_ohlcv_aligned, df_market = load_stock_and_market_data(
                stock_code=stock_code,
                end_date=TRAIN_END_DATE,
                limit=REQUIRED_KLINE_LIMIT,
                market_cache=market_cache,
                required_kline_count=REQUIRED_KLINE_COUNT,
                warmup_period=REQUIRED_WARMUP_PERIOD,
                random_sample=random_sample
            )
            
            if df_ohlcv_aligned is None:
                print(f"\r   ⚠️  股票 {stock_code} 数据加载失败", flush=True)
                continue
            
            df_normalized = gp_manager.calculate_and_normalize(df_ohlcv_aligned, market_data=df_market)
            
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in df_normalized.columns if col not in ohlcv_cols]
            
            feature_data = {
                'features_all': df_normalized[feature_cols].values,
                'closes': df_ohlcv_aligned['close'].values,
                'highs': df_ohlcv_aligned['high'].values,
                'available_codes': feature_cols,
                'feature_to_var': {feat: feat for feat in feature_cols}
            }
            
            preextracted_features[stock_code] = feature_data
            
            if (idx + 1) % 5 == 0 or (idx + 1) == len(stock_data_cache):
                print(f"\r   • 进度: {idx+1}/{len(stock_data_cache)} ({(idx+1)/len(stock_data_cache)*100:.1f}%)", end='', flush=True)
        
        except Exception as e:
            print(f"\r   ⚠️  股票 {stock_code} 特征提取失败: {e}", flush=True)
            continue
    
    print()
    
    return preextracted_features, 0, 0

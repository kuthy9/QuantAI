"""
金融产品类型定义和支持扩展
为QuantAI AutoGen系统提供多产品类型支持
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

class AssetType(str, Enum):
    """资产类型枚举"""
    EQUITY = "equity"           # 股票
    FUTURES = "futures"         # 期货
    OPTIONS = "options"         # 期权
    FOREX = "forex"            # 外汇
    COMMODITY = "commodity"     # 商品
    BOND = "bond"              # 债券
    CRYPTO = "crypto"          # 加密货币
    ETF = "etf"                # ETF
    REIT = "reit"              # REITs

class ProductType(str, Enum):
    """具体产品类型"""
    # 股票类
    STOCK = "stock"
    ETF = "etf"
    REIT = "reit"
    
    # 期货类
    EQUITY_FUTURES = "equity_futures"      # 股指期货
    COMMODITY_FUTURES = "commodity_futures" # 商品期货
    BOND_FUTURES = "bond_futures"          # 国债期货
    CURRENCY_FUTURES = "currency_futures"   # 外汇期货
    
    # 期权类
    EQUITY_OPTIONS = "equity_options"      # 股票期权
    INDEX_OPTIONS = "index_options"        # 指数期权
    COMMODITY_OPTIONS = "commodity_options" # 商品期权
    
    # 其他
    FOREX_SPOT = "forex_spot"             # 外汇现货
    CRYPTO_SPOT = "crypto_spot"           # 加密货币现货

class MarketSession(str, Enum):
    """交易时段"""
    PRE_MARKET = "pre_market"
    REGULAR = "regular"
    AFTER_HOURS = "after_hours"
    OVERNIGHT = "overnight"

@dataclass
class ProductSpecification:
    """产品规格定义"""
    symbol: str
    asset_type: AssetType
    product_type: ProductType
    exchange: str
    currency: str
    
    # 期货特有
    contract_month: Optional[str] = None
    contract_size: Optional[float] = None
    tick_size: Optional[float] = None
    margin_requirement: Optional[float] = None
    
    # 期权特有
    underlying_symbol: Optional[str] = None
    strike_price: Optional[float] = None
    expiration_date: Optional[datetime] = None
    option_type: Optional[str] = None  # "CALL", "PUT"
    
    # 外汇特有
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None
    pip_size: Optional[float] = None
    
    # 交易规则
    min_order_size: float = 1.0
    max_order_size: Optional[float] = None
    trading_sessions: List[MarketSession] = None
    
    def __post_init__(self):
        if self.trading_sessions is None:
            self.trading_sessions = [MarketSession.REGULAR]

class ProductRegistry:
    """产品注册表"""
    
    def __init__(self):
        self._products: Dict[str, ProductSpecification] = {}
        self._initialize_default_products()
    
    def _initialize_default_products(self):
        """初始化默认产品"""
        
        # 股票产品
        stocks = [
            ("AAPL", "Apple Inc."),
            ("MSFT", "Microsoft Corp."),
            ("GOOGL", "Alphabet Inc."),
            ("AMZN", "Amazon.com Inc."),
            ("TSLA", "Tesla Inc."),
            ("NVDA", "NVIDIA Corp."),
            ("META", "Meta Platforms Inc.")
        ]
        
        for symbol, name in stocks:
            self.register_product(ProductSpecification(
                symbol=symbol,
                asset_type=AssetType.EQUITY,
                product_type=ProductType.STOCK,
                exchange="NASDAQ",
                currency="USD"
            ))
        
        # ETF产品
        etfs = [
            ("SPY", "SPDR S&P 500 ETF"),
            ("QQQ", "Invesco QQQ ETF"),
            ("IWM", "iShares Russell 2000 ETF"),
            ("VTI", "Vanguard Total Stock Market ETF")
        ]
        
        for symbol, name in etfs:
            self.register_product(ProductSpecification(
                symbol=symbol,
                asset_type=AssetType.ETF,
                product_type=ProductType.ETF,
                exchange="NYSE",
                currency="USD"
            ))
        
        # 期货产品
        futures = [
            ("ES", "E-mini S&P 500", 50.0, 0.25, 0.05),  # 合约乘数, 最小变动, 保证金比例
            ("NQ", "E-mini NASDAQ-100", 20.0, 0.25, 0.05),
            ("YM", "E-mini Dow Jones", 5.0, 1.0, 0.05),
            ("GC", "Gold Futures", 100.0, 0.10, 0.10),
            ("CL", "Crude Oil Futures", 1000.0, 0.01, 0.15)
        ]
        
        for symbol, name, contract_size, tick_size, margin in futures:
            self.register_product(ProductSpecification(
                symbol=symbol,
                asset_type=AssetType.FUTURES,
                product_type=ProductType.EQUITY_FUTURES if symbol in ["ES", "NQ", "YM"] else ProductType.COMMODITY_FUTURES,
                exchange="CME",
                currency="USD",
                contract_size=contract_size,
                tick_size=tick_size,
                margin_requirement=margin,
                trading_sessions=[MarketSession.REGULAR, MarketSession.OVERNIGHT]
            ))
    
    def register_product(self, product: ProductSpecification):
        """注册产品"""
        self._products[product.symbol] = product
    
    def get_product(self, symbol: str) -> Optional[ProductSpecification]:
        """获取产品规格"""
        return self._products.get(symbol)
    
    def get_products_by_type(self, asset_type: AssetType) -> List[ProductSpecification]:
        """按资产类型获取产品"""
        return [p for p in self._products.values() if p.asset_type == asset_type]
    
    def get_tradeable_products(self, session: MarketSession = MarketSession.REGULAR) -> List[ProductSpecification]:
        """获取可交易产品"""
        return [p for p in self._products.values() if session in p.trading_sessions]

class ProductAnalyzer:
    """产品分析器"""
    
    def __init__(self, registry: ProductRegistry):
        self.registry = registry
    
    def analyze_product_suitability(self, macro_environment: Dict[str, Any]) -> Dict[AssetType, float]:
        """分析产品适用性"""
        
        economic_phase = macro_environment.get("economic_phase", "neutral")
        inflation_environment = macro_environment.get("inflation_environment", "moderate")
        market_risk = macro_environment.get("market_risk_level", "medium")
        volatility_regime = macro_environment.get("volatility_regime", "normal")
        
        suitability_scores = {}
        
        # 股票适用性
        if economic_phase == "expansion":
            suitability_scores[AssetType.EQUITY] = 0.9
        elif economic_phase == "contraction":
            suitability_scores[AssetType.EQUITY] = 0.3
        else:
            suitability_scores[AssetType.EQUITY] = 0.6
        
        # 期货适用性 (对冲和杠杆工具)
        if market_risk == "high" or volatility_regime == "high":
            suitability_scores[AssetType.FUTURES] = 0.8
        else:
            suitability_scores[AssetType.FUTURES] = 0.5
        
        # 期权适用性 (波动率策略)
        if volatility_regime == "high":
            suitability_scores[AssetType.OPTIONS] = 0.9
        elif volatility_regime == "low":
            suitability_scores[AssetType.OPTIONS] = 0.7  # 卖出期权策略
        else:
            suitability_scores[AssetType.OPTIONS] = 0.6
        
        # 商品适用性 (通胀对冲)
        if inflation_environment == "high":
            suitability_scores[AssetType.COMMODITY] = 0.8
        else:
            suitability_scores[AssetType.COMMODITY] = 0.4
        
        # ETF适用性 (分散化工具)
        suitability_scores[AssetType.ETF] = 0.7  # 通常适用
        
        # REITs适用性 (通胀对冲和收益)
        if inflation_environment in ["moderate", "high"]:
            suitability_scores[AssetType.REIT] = 0.7
        else:
            suitability_scores[AssetType.REIT] = 0.5
        
        return suitability_scores
    
    def recommend_optimal_products(
        self, 
        macro_environment: Dict[str, Any],
        max_products_per_type: int = 5
    ) -> Dict[AssetType, List[ProductSpecification]]:
        """推荐最优产品组合"""
        
        suitability_scores = self.analyze_product_suitability(macro_environment)
        recommendations = {}
        
        # 按适用性分数排序，选择最适合的资产类型
        sorted_types = sorted(suitability_scores.items(), key=lambda x: x[1], reverse=True)
        
        for asset_type, score in sorted_types:
            if score >= 0.5:  # 只推荐适用性≥50%的产品类型
                products = self.registry.get_products_by_type(asset_type)
                
                # 简单选择前N个产品 (实际应用中可以加入更复杂的筛选逻辑)
                selected_products = products[:max_products_per_type]
                
                if selected_products:
                    recommendations[asset_type] = selected_products
        
        return recommendations
    
    def calculate_portfolio_allocation(
        self,
        recommended_products: Dict[AssetType, List[ProductSpecification]],
        total_capital: float = 1000000.0
    ) -> Dict[str, Dict[str, Any]]:
        """计算投资组合配置"""
        
        allocation = {}
        
        # 简单的等权重分配策略
        total_asset_types = len(recommended_products)
        
        for asset_type, products in recommended_products.items():
            asset_weight = 1.0 / total_asset_types
            product_weight = asset_weight / len(products)
            
            for product in products:
                allocation[product.symbol] = {
                    "asset_type": asset_type,
                    "product_type": product.product_type,
                    "weight": product_weight,
                    "capital_allocation": total_capital * product_weight,
                    "exchange": product.exchange,
                    "currency": product.currency
                }
        
        return allocation

# 全局产品注册表实例
PRODUCT_REGISTRY = ProductRegistry()
PRODUCT_ANALYZER = ProductAnalyzer(PRODUCT_REGISTRY)

"""
Macro Insight Agent (A0) for the QuantAI system.

This agent analyzes macro trends, market cycles, and market regimes
to provide high-level market context for strategy generation.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from autogen_core import MessageContext
from autogen_core.models import ChatCompletionClient, UserMessage
from loguru import logger
import numpy as np

from ...core.base import AgentCapability, AgentRole, ModelCapableAgent
from ...core.messages import DataMessage, MessageType, QuantMessage, StrategyMessage


class MacroInsightAgent(ModelCapableAgent):
    """
    Macro Insight Agent (A0) - Analyzes macro trends, cycles, and market regimes.
    
    Capabilities:
    - Market regime identification (bull/bear/sideways, high/low volatility)
    - Economic cycle analysis (expansion/contraction/recession)
    - Sector rotation analysis
    - Cross-asset correlation analysis
    - Geopolitical risk assessment
    - Central bank policy impact analysis
    """
    
    def __init__(
        self,
        model_client: ChatCompletionClient,
        regime_lookback_days: int = 252,  # 1 year
        update_frequency_hours: int = 6,
        **kwargs
    ):
        super().__init__(
            role=AgentRole.MACRO_INSIGHT,
            capabilities=[
                AgentCapability.DATA_PROCESSING,
                AgentCapability.MULTIMODAL_ANALYSIS,
            ],
            model_client=model_client,
            system_message=self._get_system_message(),
            **kwargs
        )
        
        self.regime_lookback_days = regime_lookback_days
        self.update_frequency_hours = update_frequency_hours
        self._market_regimes: Dict[str, str] = {}
        self._macro_indicators: Dict[str, Any] = {}
        self._last_analysis_time: Optional[datetime] = None
    
    def _get_system_message(self) -> str:
        return """You are a Macro Insight Agent specializing in macroeconomic analysis and market regime identification.

Your responsibilities:
1. Identify current market regimes (bull/bear/sideways, high/low volatility)
2. Analyze economic cycles and their impact on asset classes
3. Monitor sector rotation patterns and leadership changes
4. Assess cross-asset correlations and risk-on/risk-off dynamics
5. Evaluate geopolitical risks and their market implications
6. Analyze central bank policies and monetary conditions

Market Regime Framework:
- Trend Regimes: Bull Market, Bear Market, Sideways/Range-bound
- Volatility Regimes: Low Volatility, High Volatility, Crisis
- Economic Regimes: Expansion, Late Cycle, Recession, Recovery
- Risk Regimes: Risk-On, Risk-Off, Transition

Key Indicators to Monitor:
- Equity indices (SPY, QQQ, IWM, VIX)
- Bond markets (TLT, HYG, LQD)
- Commodities (GLD, USO, DBA)
- Currencies (DXY, major pairs)
- Economic data (GDP, inflation, employment)
- Central bank communications

Guidelines:
- Provide clear regime classifications with confidence levels
- Identify regime transitions early using leading indicators
- Consider multiple timeframes (short, medium, long-term)
- Assess regime stability and potential change catalysts
- Provide actionable insights for strategy development
- Maintain awareness of tail risks and black swan events

Focus on providing strategic market context that enables effective strategy generation and risk management."""
    
    async def process_message(
        self, 
        message: QuantMessage, 
        ctx: MessageContext
    ) -> Optional[QuantMessage]:
        """Process incoming data for macro analysis."""
        
        if isinstance(message, DataMessage) and message.message_type == MessageType.DATA_RESPONSE:
            # Process fused signals for macro analysis
            if message.data_type == "fused_signals":
                await self._analyze_macro_signals(message)
                
                # Generate macro insights
                insights = await self._generate_macro_insights(message.symbols)
                
                # Create strategy message with macro context
                response = StrategyMessage(
                    message_type=MessageType.STRATEGY_GENERATION,
                    sender_id=self.agent_id,
                    strategy_id=f"macro_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    strategy_description="Macro market context and regime analysis",
                    strategy_parameters=insights,
                    session_id=message.session_id,
                    correlation_id=message.correlation_id,
                )
                
                logger.info("Generated macro insights")
                return response
        
        return None
    
    async def _analyze_macro_signals(self, message: DataMessage):
        """Analyze fused signals for macro patterns."""
        signals_data = message.data_payload
        
        if not signals_data:
            return
        
        # Extract macro-relevant signals
        macro_signals = {}
        
        for symbol, signal_data in signals_data.items():
            if symbol == "fusion_metadata":
                continue
                
            if isinstance(signal_data, dict) and "final_signal" in signal_data:
                macro_signals[symbol] = {
                    "signal": signal_data["final_signal"],
                    "confidence": signal_data.get("confidence", 0.0),
                    "components": signal_data.get("component_signals", {}),
                }
        
        # Update macro indicators
        await self._update_macro_indicators(macro_signals)
        
        # Identify market regimes
        await self._identify_market_regimes(macro_signals)
    
    async def _update_macro_indicators(self, signals: Dict[str, Any]):
        """Update macro economic indicators."""
        
        # Market breadth indicators
        if "SPY" in signals and "QQQ" in signals and "IWM" in signals:
            spy_signal = signals["SPY"]["signal"]
            qqq_signal = signals["QQQ"]["signal"]
            iwm_signal = signals["IWM"]["signal"]
            
            # Calculate market breadth
            positive_signals = sum(1 for s in [spy_signal, qqq_signal, iwm_signal] if s > 0)
            market_breadth = positive_signals / 3.0
            
            self._macro_indicators["market_breadth"] = market_breadth
        
        # Risk-on/Risk-off sentiment
        risk_on_score = 0.0
        risk_indicators = 0
        
        for symbol in ["SPY", "QQQ", "IWM"]:  # Risk-on assets
            if symbol in signals:
                risk_on_score += signals[symbol]["signal"]
                risk_indicators += 1
        
        if risk_indicators > 0:
            self._macro_indicators["risk_sentiment"] = risk_on_score / risk_indicators
        
        # Update timestamp
        self._macro_indicators["last_updated"] = datetime.now().isoformat()
    
    async def _identify_market_regimes(self, signals: Dict[str, Any]):
        """Identify current market regimes."""
        
        # Trend regime identification
        trend_regime = await self._identify_trend_regime(signals)
        self._market_regimes["trend"] = trend_regime
        
        # Volatility regime identification  
        volatility_regime = await self._identify_volatility_regime(signals)
        self._market_regimes["volatility"] = volatility_regime
        
        # Risk regime identification
        risk_regime = await self._identify_risk_regime(signals)
        self._market_regimes["risk"] = risk_regime
        
        logger.info(f"Market regimes: {self._market_regimes}")
    
    async def _identify_trend_regime(self, signals: Dict[str, Any]) -> str:
        """Identify the current trend regime."""
        
        # Use SPY as primary market indicator
        if "SPY" not in signals:
            return "unknown"
        
        spy_signal = signals["SPY"]["signal"]
        spy_confidence = signals["SPY"]["confidence"]
        
        # Simple regime classification
        if spy_signal > 0.3 and spy_confidence > 0.7:
            return "bull_market"
        elif spy_signal < -0.3 and spy_confidence > 0.7:
            return "bear_market"
        else:
            return "sideways"
    
    async def _identify_volatility_regime(self, signals: Dict[str, Any]) -> str:
        """Identify the current volatility regime."""
        
        # This would typically use VIX data
        # For now, use signal variance as proxy
        signal_values = []
        for symbol_data in signals.values():
            if isinstance(symbol_data, dict) and "signal" in symbol_data:
                signal_values.append(symbol_data["signal"])
        
        if signal_values:
            signal_variance = np.var(signal_values)
            if signal_variance > 0.5:
                return "high_volatility"
            elif signal_variance < 0.1:
                return "low_volatility"
            else:
                return "normal_volatility"
        
        return "unknown"
    
    async def _identify_risk_regime(self, signals: Dict[str, Any]) -> str:
        """Identify the current risk regime."""
        
        risk_sentiment = self._macro_indicators.get("risk_sentiment", 0.0)
        
        if risk_sentiment > 0.2:
            return "risk_on"
        elif risk_sentiment < -0.2:
            return "risk_off"
        else:
            return "neutral"
    
    async def _generate_macro_insights(self, symbols: List[str]) -> Dict[str, Any]:
        """Generate comprehensive macro insights using LLM analysis."""
        
        # Prepare context for LLM
        context = {
            "market_regimes": self._market_regimes,
            "macro_indicators": self._macro_indicators,
            "analysis_timestamp": datetime.now().isoformat(),
        }
        
        prompt = f"""Analyze the current macro market environment based on the following data:

Market Regimes:
{json.dumps(self._market_regimes, indent=2)}

Macro Indicators:
{json.dumps(self._macro_indicators, indent=2)}

Please provide a comprehensive macro analysis including:

1. Current Market Environment Assessment
   - Overall market regime and trend direction
   - Volatility environment and risk conditions
   - Key market drivers and themes

2. Economic Cycle Analysis
   - Current phase of economic cycle
   - Leading indicators and potential inflection points
   - Central bank policy implications

3. Cross-Asset Analysis
   - Equity market leadership and sector rotation
   - Bond market signals and yield curve dynamics
   - Currency and commodity trends

4. Risk Assessment
   - Key risks and potential catalysts
   - Tail risk scenarios to monitor
   - Risk-on vs risk-off dynamics

5. Strategic Implications
   - Recommended asset allocation themes
   - Sector and style preferences
   - Hedging considerations

6. Regime Change Indicators
   - Early warning signals to monitor
   - Potential regime transition catalysts
   - Timeline for potential changes

Respond in JSON format with clear, actionable insights:
{{
    "market_environment": {{
        "overall_assessment": "description",
        "trend_direction": "bullish/bearish/neutral",
        "volatility_outlook": "low/normal/high",
        "confidence_level": 0.0-1.0
    }},
    "economic_cycle": {{
        "current_phase": "expansion/late_cycle/recession/recovery",
        "leading_indicators": ["indicator1", "indicator2"],
        "policy_implications": "description"
    }},
    "cross_asset_analysis": {{
        "equity_leadership": "growth/value/small/large",
        "bond_signals": "description",
        "currency_trends": "description"
    }},
    "risk_assessment": {{
        "key_risks": ["risk1", "risk2"],
        "tail_risks": ["tail_risk1", "tail_risk2"],
        "risk_level": "low/medium/high"
    }},
    "strategic_implications": {{
        "asset_allocation": "description",
        "sector_preferences": ["sector1", "sector2"],
        "hedging_needs": "description"
    }},
    "regime_change_indicators": {{
        "warning_signals": ["signal1", "signal2"],
        "potential_catalysts": ["catalyst1", "catalyst2"],
        "timeline": "short/medium/long-term"
    }}
}}"""
        
        try:
            response = await self._call_model([UserMessage(content=prompt, source="user")])
            macro_analysis = json.loads(response)
            
            # Add metadata
            macro_analysis["metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "agent_id": self.agent_id,
                "symbols_analyzed": symbols,
                "regime_data": self._market_regimes,
                "indicators_data": self._macro_indicators,
            }
            
            return macro_analysis
            
        except Exception as e:
            logger.error(f"Error generating macro insights: {e}")
            return {
                "error": str(e),
                "market_regimes": self._market_regimes,
                "macro_indicators": self._macro_indicators,
            }
    
    async def get_current_regime(self) -> Dict[str, str]:
        """Get the current market regime assessment."""
        return self._market_regimes.copy()
    
    async def get_macro_indicators(self) -> Dict[str, Any]:
        """Get the current macro indicators."""
        return self._macro_indicators.copy()
    
    async def start_periodic_analysis(self):
        """Start periodic macro analysis."""
        logger.info(f"Starting periodic macro analysis every {self.update_frequency_hours} hours")
        
        while True:
            try:
                # Check if we need to update analysis
                if (self._last_analysis_time is None or 
                    datetime.now() - self._last_analysis_time > timedelta(hours=self.update_frequency_hours)):
                    
                    # Request latest fused signals
                    await self._request_latest_data()
                    self._last_analysis_time = datetime.now()
                
                # Wait for next check
                await asyncio.sleep(3600)  # Check every hour
                
            except Exception as e:
                logger.error(f"Error in periodic macro analysis: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying
    
    async def _request_latest_data(self):
        """Request latest fused signals for analysis."""
        # This would typically send a data request message
        # For now, we'll just log the intent
        logger.info("Requesting latest fused signals for macro analysis")

    async def autonomous_investment_analysis(self) -> Dict[str, Any]:
        """
        自主投资决策分析 - 无需用户输入，基于宏观分析推荐投资标的
        """
        logger.info("Starting autonomous investment analysis")

        # 1. 收集宏观经济数据
        macro_data = await self._collect_macro_economic_data()

        # 2. 分析经济周期和市场环境
        economic_analysis = await self._analyze_economic_environment(macro_data)

        # 3. 推荐行业板块
        sector_recommendations = await self._recommend_sectors(economic_analysis)

        # 4. 筛选具体投资标的
        investment_targets = await self._select_investment_targets(sector_recommendations)

        # 5. 生成投资决策报告
        investment_decision = {
            "analysis_timestamp": datetime.now().isoformat(),
            "macro_environment": economic_analysis,
            "recommended_sectors": sector_recommendations,
            "investment_targets": investment_targets,
            "decision_confidence": self._calculate_decision_confidence(economic_analysis, sector_recommendations),
            "risk_assessment": await self._assess_investment_risks(investment_targets),
            "allocation_strategy": await self._generate_allocation_strategy(investment_targets)
        }

        logger.info(f"Autonomous investment analysis completed: {len(investment_targets)} targets identified")
        return investment_decision

    async def autonomous_multi_product_analysis(self) -> Dict[str, Any]:
        """
        跨产品类型的自主投资决策分析
        支持股票、期货、期权等多种金融产品的智能配置
        """
        try:
            from quantai.core.financial_products import PRODUCT_ANALYZER, AssetType, ProductType

            # 1. 宏观环境分析
            macro_analysis = await self._analyze_macro_environment()

            # 2. 产品类型选择
            optimal_products = await self._select_optimal_product_types(macro_analysis)

            # 3. 具体产品推荐
            investment_recommendations = {}

            for asset_type in optimal_products:
                if asset_type == AssetType.EQUITY:
                    recommendations = await self._recommend_equity_products(macro_analysis)
                elif asset_type == AssetType.FUTURES:
                    recommendations = await self._recommend_futures_products(macro_analysis)
                elif asset_type == AssetType.OPTIONS:
                    recommendations = await self._recommend_options_strategies(macro_analysis)
                elif asset_type == AssetType.ETF:
                    recommendations = await self._recommend_etf_products(macro_analysis)
                elif asset_type == AssetType.COMMODITY:
                    recommendations = await self._recommend_commodity_products(macro_analysis)
                else:
                    recommendations = []

                if recommendations:
                    investment_recommendations[asset_type.value] = recommendations

            # 4. 生成多产品投资组合配置
            portfolio_allocation = await self._generate_multi_product_allocation(investment_recommendations)

            # 5. 多产品风险评估
            risk_assessment = await self._assess_multi_product_risks(investment_recommendations)

            return {
                "analysis_type": "multi_product_autonomous_decision",
                "macro_environment": macro_analysis,
                "optimal_product_types": [pt.value for pt in optimal_products],
                "investment_recommendations": investment_recommendations,
                "portfolio_allocation": portfolio_allocation,
                "risk_assessment": risk_assessment,
                "decision_confidence": self._calculate_multi_product_confidence(investment_recommendations),
                "implementation_timeline": "immediate_to_4_weeks",
                "rebalancing_frequency": "monthly"
            }

        except ImportError:
            # Fallback to original equity-only analysis
            return await self.autonomous_investment_analysis()

    async def _select_optimal_product_types(self, macro_analysis: Dict[str, Any]) -> List:
        """基于宏观分析选择最优产品类型"""
        try:
            from quantai.core.financial_products import AssetType

            economic_phase = macro_analysis.get("economic_phase", "neutral")
            inflation_environment = macro_analysis.get("inflation_environment", "moderate")
            market_risk = macro_analysis.get("market_risk_level", "medium")
            volatility_regime = macro_analysis.get("volatility_regime", "normal")

            optimal_products = []

            # 基于经济周期选择产品类型
            if economic_phase == "expansion":
                optimal_products.extend([AssetType.EQUITY, AssetType.ETF])
                if market_risk == "low":
                    optimal_products.append(AssetType.OPTIONS)  # 卖出期权策略

            elif economic_phase == "contraction":
                optimal_products.extend([AssetType.FUTURES])  # 对冲工具
                optimal_products.append(AssetType.OPTIONS)  # 保护性期权

            else:  # neutral
                optimal_products.extend([AssetType.EQUITY, AssetType.ETF])

            # 基于通胀环境调整
            if inflation_environment in ["high", "rising"]:
                optimal_products.extend([AssetType.COMMODITY, AssetType.REIT])

            # 基于波动率环境
            if volatility_regime == "high":
                optimal_products.append(AssetType.FUTURES)  # 对冲工具
                optimal_products.append(AssetType.OPTIONS)  # 波动率策略

            # 去重并确保至少有股票
            unique_products = list(set(optimal_products))
            if not unique_products:
                unique_products = [AssetType.EQUITY]

            return unique_products

        except ImportError:
            # Fallback to equity only
            return ["equity"]

    async def _recommend_futures_products(self, macro_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """推荐期货产品"""

        economic_phase = macro_analysis.get("economic_phase", "neutral")
        market_risk = macro_analysis.get("market_risk_level", "medium")

        futures_recommendations = []

        # 股指期货推荐
        if economic_phase in ["expansion", "neutral"]:
            futures_recommendations.extend([
                {
                    "symbol": "ES",
                    "name": "E-mini S&P 500 Futures",
                    "product_type": "equity_futures",
                    "strategy": "trend_following",
                    "leverage": 3.0,
                    "allocation_weight": 0.15,
                    "risk_level": "medium-high"
                },
                {
                    "symbol": "NQ",
                    "name": "E-mini NASDAQ-100 Futures",
                    "product_type": "equity_futures",
                    "strategy": "momentum",
                    "leverage": 2.5,
                    "allocation_weight": 0.10,
                    "risk_level": "high"
                }
            ])

        # 商品期货推荐
        if macro_analysis.get("inflation_environment") in ["high", "rising"]:
            futures_recommendations.extend([
                {
                    "symbol": "GC",
                    "name": "Gold Futures",
                    "product_type": "commodity_futures",
                    "strategy": "inflation_hedge",
                    "leverage": 2.0,
                    "allocation_weight": 0.08,
                    "risk_level": "medium"
                },
                {
                    "symbol": "CL",
                    "name": "Crude Oil Futures",
                    "product_type": "commodity_futures",
                    "strategy": "commodity_momentum",
                    "leverage": 2.5,
                    "allocation_weight": 0.05,
                    "risk_level": "high"
                }
            ])

        return futures_recommendations

    async def _recommend_options_strategies(self, macro_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """推荐期权策略"""

        volatility_regime = macro_analysis.get("volatility_regime", "normal")
        market_risk = macro_analysis.get("market_risk_level", "medium")

        options_recommendations = []

        if volatility_regime == "high":
            # 高波动率环境：卖出期权策略
            options_recommendations.extend([
                {
                    "strategy_name": "Iron Condor SPY",
                    "underlying": "SPY",
                    "strategy_type": "iron_condor",
                    "volatility_strategy": "short_volatility",
                    "expected_return": 0.02,
                    "max_risk": 0.08,
                    "allocation_weight": 0.05
                },
                {
                    "strategy_name": "Covered Call Portfolio",
                    "underlying": "QQQ",
                    "strategy_type": "covered_call",
                    "income_enhancement": 0.015,
                    "downside_protection": 0.03,
                    "allocation_weight": 0.10
                }
            ])

        elif volatility_regime == "low":
            # 低波动率环境：买入期权策略
            options_recommendations.extend([
                {
                    "strategy_name": "Long Straddle",
                    "underlying": "SPY",
                    "strategy_type": "long_straddle",
                    "volatility_strategy": "long_volatility",
                    "breakeven_range": 0.05,
                    "allocation_weight": 0.03
                }
            ])

        if market_risk == "high":
            # 高风险环境：保护性期权
            options_recommendations.append({
                "strategy_name": "Portfolio Protection",
                "underlying": "SPY",
                "strategy_type": "protective_put",
                "protection_level": 0.10,
                "cost": 0.02,
                "allocation_weight": 0.05
            })

        return options_recommendations

    async def _recommend_etf_products(self, macro_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """推荐ETF产品"""

        etf_recommendations = [
            {
                "symbol": "SPY",
                "name": "SPDR S&P 500 ETF",
                "category": "large_cap_equity",
                "allocation_weight": 0.25,
                "risk_level": "medium"
            },
            {
                "symbol": "QQQ",
                "name": "Invesco QQQ ETF",
                "category": "technology_growth",
                "allocation_weight": 0.15,
                "risk_level": "medium-high"
            },
            {
                "symbol": "IWM",
                "name": "iShares Russell 2000 ETF",
                "category": "small_cap_equity",
                "allocation_weight": 0.10,
                "risk_level": "high"
            }
        ]

        return etf_recommendations

    async def _recommend_commodity_products(self, macro_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """推荐商品产品"""

        commodity_recommendations = []

        if macro_analysis.get("inflation_environment") in ["high", "rising"]:
            commodity_recommendations.extend([
                {
                    "symbol": "GLD",
                    "name": "SPDR Gold Shares",
                    "category": "precious_metals",
                    "allocation_weight": 0.08,
                    "inflation_hedge": True
                },
                {
                    "symbol": "USO",
                    "name": "United States Oil Fund",
                    "category": "energy",
                    "allocation_weight": 0.05,
                    "inflation_hedge": True
                }
            ])

        return commodity_recommendations

    async def _generate_multi_product_allocation(self, investment_recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """生成多产品投资组合配置"""

        total_allocation = {}
        allocation_by_type = {}

        for asset_type, products in investment_recommendations.items():
            type_allocation = 0.0

            for product in products:
                weight = product.get("allocation_weight", 0.05)
                symbol = product.get("symbol", product.get("strategy_name", "unknown"))

                total_allocation[symbol] = {
                    "asset_type": asset_type,
                    "weight": weight,
                    "product_info": product
                }

                type_allocation += weight

            allocation_by_type[asset_type] = type_allocation

        return {
            "total_allocation": total_allocation,
            "allocation_by_type": allocation_by_type,
            "diversification_score": len(investment_recommendations) / 5.0,  # 最多5种资产类型
            "total_weight": sum(allocation_by_type.values())
        }

    async def _assess_multi_product_risks(self, investment_recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """评估多产品投资组合风险"""

        risk_metrics = {
            "overall_risk_level": "medium",
            "leverage_exposure": 0.0,
            "volatility_exposure": 0.0,
            "concentration_risk": "low",
            "liquidity_risk": "low"
        }

        # 计算杠杆暴露
        total_leverage = 0.0
        leveraged_products = 0

        for asset_type, products in investment_recommendations.items():
            for product in products:
                leverage = product.get("leverage", 1.0)
                weight = product.get("allocation_weight", 0.0)

                if leverage > 1.0:
                    total_leverage += leverage * weight
                    leveraged_products += 1

        risk_metrics["leverage_exposure"] = total_leverage

        # 评估整体风险水平
        if total_leverage > 0.5:
            risk_metrics["overall_risk_level"] = "high"
        elif total_leverage > 0.2:
            risk_metrics["overall_risk_level"] = "medium-high"

        # 评估集中度风险
        if len(investment_recommendations) < 2:
            risk_metrics["concentration_risk"] = "high"
        elif len(investment_recommendations) < 3:
            risk_metrics["concentration_risk"] = "medium"

        return risk_metrics

    def _calculate_multi_product_confidence(self, investment_recommendations: Dict[str, Any]) -> float:
        """计算多产品决策置信度"""

        # 基于产品多样性和配置合理性计算置信度
        num_asset_types = len(investment_recommendations)
        total_products = sum(len(products) for products in investment_recommendations.values())

        # 多样性得分
        diversity_score = min(num_asset_types / 4.0, 1.0)  # 最多4种资产类型为满分

        # 产品数量得分
        quantity_score = min(total_products / 15.0, 1.0)  # 15个产品为满分

        # 综合置信度
        confidence = 0.7 + 0.2 * diversity_score + 0.1 * quantity_score

        return min(confidence, 0.95)  # 最高95%置信度

    async def _collect_macro_economic_data(self) -> Dict[str, Any]:
        """收集宏观经济数据"""
        # 模拟收集各种宏观经济指标
        macro_data = {
            "gdp_growth": 2.1,  # GDP增长率
            "inflation_rate": 3.2,  # 通胀率
            "unemployment_rate": 3.7,  # 失业率
            "interest_rates": {
                "fed_funds_rate": 5.25,
                "10y_treasury": 4.5,
                "yield_curve_slope": -0.75  # 收益率曲线倒挂
            },
            "market_indicators": {
                "vix": 18.5,  # 恐慌指数
                "dollar_index": 103.2,  # 美元指数
                "oil_price": 78.5,  # 原油价格
                "gold_price": 2020.0  # 黄金价格
            },
            "economic_sentiment": {
                "consumer_confidence": 102.3,
                "business_confidence": 98.7,
                "manufacturing_pmi": 49.2  # 制造业PMI
            },
            "geopolitical_factors": {
                "trade_tensions": "moderate",
                "geopolitical_risk": "elevated",
                "policy_uncertainty": "high"
            }
        }

        logger.info("Macro economic data collected")
        return macro_data

    async def _analyze_economic_environment(self, macro_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析经济环境"""

        # 经济周期判断
        gdp_growth = macro_data["gdp_growth"]
        inflation = macro_data["inflation_rate"]
        unemployment = macro_data["unemployment_rate"]

        if gdp_growth > 2.5 and unemployment < 4.0:
            economic_phase = "expansion"
            growth_outlook = "positive"
        elif gdp_growth < 1.0 or unemployment > 6.0:
            economic_phase = "contraction"
            growth_outlook = "negative"
        else:
            economic_phase = "transition"
            growth_outlook = "neutral"

        # 通胀环境分析
        if inflation > 4.0:
            inflation_environment = "high_inflation"
            monetary_policy_bias = "hawkish"
        elif inflation < 2.0:
            inflation_environment = "low_inflation"
            monetary_policy_bias = "dovish"
        else:
            inflation_environment = "moderate_inflation"
            monetary_policy_bias = "neutral"

        # 市场风险评估
        vix = macro_data["market_indicators"]["vix"]
        if vix > 25:
            market_risk = "high"
        elif vix < 15:
            market_risk = "low"
        else:
            market_risk = "moderate"

        analysis = {
            "economic_phase": economic_phase,
            "growth_outlook": growth_outlook,
            "inflation_environment": inflation_environment,
            "monetary_policy_bias": monetary_policy_bias,
            "market_risk_level": market_risk,
            "yield_curve_signal": "inverted" if macro_data["interest_rates"]["yield_curve_slope"] < 0 else "normal",
            "dollar_strength": "strong" if macro_data["market_indicators"]["dollar_index"] > 100 else "weak",
            "commodity_trend": "bullish" if macro_data["market_indicators"]["oil_price"] > 75 else "bearish"
        }

        logger.info(f"Economic environment analysis: {analysis['economic_phase']}, {analysis['growth_outlook']}")
        return analysis

    async def _recommend_sectors(self, economic_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于经济分析推荐行业板块"""

        sector_recommendations = []

        economic_phase = economic_analysis["economic_phase"]
        inflation_env = economic_analysis["inflation_environment"]
        market_risk = economic_analysis["market_risk_level"]

        # 基于经济周期的板块轮动策略
        if economic_phase == "expansion":
            # 扩张期：偏好周期性行业
            sector_recommendations.extend([
                {
                    "sector": "Technology",
                    "rationale": "经济扩张期技术股表现强劲",
                    "confidence": 0.85,
                    "allocation_weight": 0.25,
                    "key_themes": ["AI革命", "云计算增长", "数字化转型"]
                },
                {
                    "sector": "Consumer Discretionary",
                    "rationale": "消费者支出增长支撑可选消费",
                    "confidence": 0.80,
                    "allocation_weight": 0.20,
                    "key_themes": ["消费复苏", "电商增长", "旅游恢复"]
                },
                {
                    "sector": "Financials",
                    "rationale": "利率上升环境有利于银行业",
                    "confidence": 0.75,
                    "allocation_weight": 0.15,
                    "key_themes": ["净息差扩大", "信贷需求增长", "监管放松"]
                }
            ])

        elif economic_phase == "contraction":
            # 收缩期：偏好防御性行业
            sector_recommendations.extend([
                {
                    "sector": "Healthcare",
                    "rationale": "医疗需求刚性，防御性强",
                    "confidence": 0.90,
                    "allocation_weight": 0.30,
                    "key_themes": ["人口老龄化", "创新药物", "医疗技术进步"]
                },
                {
                    "sector": "Consumer Staples",
                    "rationale": "必需消费品需求稳定",
                    "confidence": 0.85,
                    "allocation_weight": 0.25,
                    "key_themes": ["刚性需求", "品牌价值", "定价能力"]
                },
                {
                    "sector": "Utilities",
                    "rationale": "公用事业提供稳定股息",
                    "confidence": 0.80,
                    "allocation_weight": 0.20,
                    "key_themes": ["稳定现金流", "高股息", "能源转型"]
                }
            ])

        else:  # transition
            # 过渡期：平衡配置
            sector_recommendations.extend([
                {
                    "sector": "Technology",
                    "rationale": "长期增长趋势不变",
                    "confidence": 0.75,
                    "allocation_weight": 0.20,
                    "key_themes": ["结构性增长", "创新驱动", "效率提升"]
                },
                {
                    "sector": "Healthcare",
                    "rationale": "防御性与增长性兼备",
                    "confidence": 0.80,
                    "allocation_weight": 0.20,
                    "key_themes": ["稳健增长", "创新管线", "并购活动"]
                },
                {
                    "sector": "Energy",
                    "rationale": "能源转型和地缘政治支撑",
                    "confidence": 0.70,
                    "allocation_weight": 0.15,
                    "key_themes": ["能源安全", "绿色转型", "资本纪律"]
                }
            ])

        # 基于通胀环境调整
        if inflation_env == "high_inflation":
            # 高通胀环境下增加抗通胀板块
            sector_recommendations.append({
                "sector": "Real Estate",
                "rationale": "房地产是通胀对冲工具",
                "confidence": 0.75,
                "allocation_weight": 0.10,
                "key_themes": ["通胀对冲", "租金增长", "资产升值"]
            })

        logger.info(f"Recommended {len(sector_recommendations)} sectors based on economic analysis")
        return sector_recommendations

    async def _select_investment_targets(self, sector_recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """基于板块推荐选择具体投资标的"""

        investment_targets = []

        # 各板块的优质标的映射
        sector_stocks = {
            "Technology": [
                {"symbol": "AAPL", "name": "Apple Inc.", "market_cap": "large", "quality_score": 0.95},
                {"symbol": "MSFT", "name": "Microsoft Corp.", "market_cap": "large", "quality_score": 0.93},
                {"symbol": "GOOGL", "name": "Alphabet Inc.", "market_cap": "large", "quality_score": 0.90},
                {"symbol": "NVDA", "name": "NVIDIA Corp.", "market_cap": "large", "quality_score": 0.88},
                {"symbol": "META", "name": "Meta Platforms", "market_cap": "large", "quality_score": 0.85}
            ],
            "Healthcare": [
                {"symbol": "JNJ", "name": "Johnson & Johnson", "market_cap": "large", "quality_score": 0.92},
                {"symbol": "UNH", "name": "UnitedHealth Group", "market_cap": "large", "quality_score": 0.90},
                {"symbol": "PFE", "name": "Pfizer Inc.", "market_cap": "large", "quality_score": 0.85},
                {"symbol": "ABBV", "name": "AbbVie Inc.", "market_cap": "large", "quality_score": 0.88},
                {"symbol": "TMO", "name": "Thermo Fisher Scientific", "market_cap": "large", "quality_score": 0.87}
            ],
            "Consumer Discretionary": [
                {"symbol": "AMZN", "name": "Amazon.com Inc.", "market_cap": "large", "quality_score": 0.89},
                {"symbol": "TSLA", "name": "Tesla Inc.", "market_cap": "large", "quality_score": 0.82},
                {"symbol": "HD", "name": "Home Depot Inc.", "market_cap": "large", "quality_score": 0.88},
                {"symbol": "MCD", "name": "McDonald's Corp.", "market_cap": "large", "quality_score": 0.86},
                {"symbol": "NKE", "name": "Nike Inc.", "market_cap": "large", "quality_score": 0.84}
            ],
            "Financials": [
                {"symbol": "JPM", "name": "JPMorgan Chase & Co.", "market_cap": "large", "quality_score": 0.90},
                {"symbol": "BAC", "name": "Bank of America Corp.", "market_cap": "large", "quality_score": 0.85},
                {"symbol": "WFC", "name": "Wells Fargo & Co.", "market_cap": "large", "quality_score": 0.80},
                {"symbol": "GS", "name": "Goldman Sachs Group", "market_cap": "large", "quality_score": 0.87},
                {"symbol": "MS", "name": "Morgan Stanley", "market_cap": "large", "quality_score": 0.86}
            ],
            "Consumer Staples": [
                {"symbol": "PG", "name": "Procter & Gamble Co.", "market_cap": "large", "quality_score": 0.90},
                {"symbol": "KO", "name": "Coca-Cola Co.", "market_cap": "large", "quality_score": 0.88},
                {"symbol": "PEP", "name": "PepsiCo Inc.", "market_cap": "large", "quality_score": 0.87},
                {"symbol": "WMT", "name": "Walmart Inc.", "market_cap": "large", "quality_score": 0.85},
                {"symbol": "COST", "name": "Costco Wholesale Corp.", "market_cap": "large", "quality_score": 0.89}
            ],
            "Energy": [
                {"symbol": "XOM", "name": "Exxon Mobil Corp.", "market_cap": "large", "quality_score": 0.82},
                {"symbol": "CVX", "name": "Chevron Corp.", "market_cap": "large", "quality_score": 0.85},
                {"symbol": "COP", "name": "ConocoPhillips", "market_cap": "large", "quality_score": 0.83},
                {"symbol": "EOG", "name": "EOG Resources Inc.", "market_cap": "large", "quality_score": 0.80},
                {"symbol": "SLB", "name": "Schlumberger Ltd.", "market_cap": "large", "quality_score": 0.78}
            ],
            "Utilities": [
                {"symbol": "NEE", "name": "NextEra Energy Inc.", "market_cap": "large", "quality_score": 0.88},
                {"symbol": "DUK", "name": "Duke Energy Corp.", "market_cap": "large", "quality_score": 0.85},
                {"symbol": "SO", "name": "Southern Co.", "market_cap": "large", "quality_score": 0.83},
                {"symbol": "D", "name": "Dominion Energy Inc.", "market_cap": "large", "quality_score": 0.82},
                {"symbol": "AEP", "name": "American Electric Power", "market_cap": "large", "quality_score": 0.84}
            ],
            "Real Estate": [
                {"symbol": "AMT", "name": "American Tower Corp.", "market_cap": "large", "quality_score": 0.87},
                {"symbol": "PLD", "name": "Prologis Inc.", "market_cap": "large", "quality_score": 0.85},
                {"symbol": "CCI", "name": "Crown Castle Inc.", "market_cap": "large", "quality_score": 0.83},
                {"symbol": "EQIX", "name": "Equinix Inc.", "market_cap": "large", "quality_score": 0.86},
                {"symbol": "SPG", "name": "Simon Property Group", "market_cap": "large", "quality_score": 0.80}
            ]
        }

        # 为每个推荐板块选择最优标的
        for sector_rec in sector_recommendations:
            sector_name = sector_rec["sector"]
            allocation_weight = sector_rec["allocation_weight"]
            sector_confidence = sector_rec["confidence"]

            if sector_name in sector_stocks:
                available_stocks = sector_stocks[sector_name]

                # 根据质量分数和配置权重选择标的
                num_stocks_to_select = min(3, len(available_stocks))  # 每个板块最多选3只股票

                # 按质量分数排序并选择前几名
                sorted_stocks = sorted(available_stocks, key=lambda x: x["quality_score"], reverse=True)
                selected_stocks = sorted_stocks[:num_stocks_to_select]

                # 为每只股票分配权重
                stock_weight = allocation_weight / num_stocks_to_select

                for stock in selected_stocks:
                    investment_target = {
                        "symbol": stock["symbol"],
                        "company_name": stock["name"],
                        "sector": sector_name,
                        "allocation_weight": stock_weight,
                        "quality_score": stock["quality_score"],
                        "selection_rationale": sector_rec["rationale"],
                        "confidence": sector_confidence * stock["quality_score"],  # 综合置信度
                        "market_cap": stock["market_cap"],
                        "investment_themes": sector_rec["key_themes"],
                        "expected_return": self._estimate_expected_return(stock, sector_rec),
                        "risk_level": self._assess_stock_risk(stock, sector_name)
                    }

                    investment_targets.append(investment_target)

        # 按置信度排序
        investment_targets.sort(key=lambda x: x["confidence"], reverse=True)

        logger.info(f"Selected {len(investment_targets)} investment targets across {len(sector_recommendations)} sectors")
        return investment_targets

    def _estimate_expected_return(self, stock: Dict[str, Any], sector_rec: Dict[str, Any]) -> float:
        """估算预期收益率"""
        base_return = 0.08  # 基础市场收益率

        # 基于质量分数调整
        quality_premium = (stock["quality_score"] - 0.8) * 0.1

        # 基于板块配置权重调整（权重越高，预期收益越高）
        sector_premium = sector_rec["allocation_weight"] * 0.05

        # 基于置信度调整
        confidence_premium = (sector_rec["confidence"] - 0.7) * 0.03

        expected_return = base_return + quality_premium + sector_premium + confidence_premium
        return round(expected_return, 3)

    def _assess_stock_risk(self, stock: Dict[str, Any], sector: str) -> str:
        """评估股票风险水平"""
        # 基于板块和质量分数评估风险
        defensive_sectors = ["Healthcare", "Consumer Staples", "Utilities"]
        cyclical_sectors = ["Technology", "Consumer Discretionary", "Financials", "Energy"]

        if sector in defensive_sectors:
            base_risk = "low"
        elif sector in cyclical_sectors:
            base_risk = "medium"
        else:
            base_risk = "medium"

        # 基于质量分数调整风险
        if stock["quality_score"] > 0.9:
            if base_risk == "medium":
                return "low-medium"
            else:
                return base_risk
        elif stock["quality_score"] < 0.8:
            if base_risk == "low":
                return "medium"
            else:
                return "high"

        return base_risk

    def _calculate_decision_confidence(self, economic_analysis: Dict[str, Any], sector_recommendations: List[Dict[str, Any]]) -> float:
        """计算整体决策置信度"""

        # 基于经济分析的确定性
        economic_certainty = 0.8  # 基础确定性

        # 基于市场风险调整
        market_risk = economic_analysis["market_risk_level"]
        if market_risk == "low":
            risk_adjustment = 0.1
        elif market_risk == "high":
            risk_adjustment = -0.2
        else:
            risk_adjustment = 0.0

        # 基于板块推荐的平均置信度
        if sector_recommendations:
            avg_sector_confidence = sum(rec["confidence"] for rec in sector_recommendations) / len(sector_recommendations)
        else:
            avg_sector_confidence = 0.5

        # 综合置信度
        overall_confidence = (economic_certainty + risk_adjustment + avg_sector_confidence) / 2
        return round(max(0.0, min(1.0, overall_confidence)), 2)

    async def _assess_investment_risks(self, investment_targets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估投资组合风险"""

        if not investment_targets:
            return {"error": "No investment targets to assess"}

        # 计算组合风险指标
        total_allocation = sum(target["allocation_weight"] for target in investment_targets)

        # 板块集中度风险
        sector_allocation = {}
        for target in investment_targets:
            sector = target["sector"]
            sector_allocation[sector] = sector_allocation.get(sector, 0) + target["allocation_weight"]

        max_sector_weight = max(sector_allocation.values()) if sector_allocation else 0
        concentration_risk = "high" if max_sector_weight > 0.4 else "medium" if max_sector_weight > 0.25 else "low"

        # 质量风险评估
        avg_quality = sum(target["quality_score"] for target in investment_targets) / len(investment_targets)
        quality_risk = "low" if avg_quality > 0.85 else "medium" if avg_quality > 0.8 else "high"

        # 市值风险（目前都是大盘股，风险较低）
        market_cap_risk = "low"  # 所有标的都是大盘股

        risk_assessment = {
            "overall_risk_level": "medium",  # 综合风险水平
            "concentration_risk": concentration_risk,
            "quality_risk": quality_risk,
            "market_cap_risk": market_cap_risk,
            "sector_diversification": len(sector_allocation),
            "max_sector_weight": round(max_sector_weight, 2),
            "portfolio_quality_score": round(avg_quality, 2),
            "risk_factors": [
                "市场波动风险",
                "板块轮动风险",
                "个股特定风险",
                "宏观经济风险"
            ],
            "mitigation_strategies": [
                "定期再平衡",
                "止损策略",
                "分批建仓",
                "动态调整"
            ]
        }

        return risk_assessment

    async def _generate_allocation_strategy(self, investment_targets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成配置策略"""

        if not investment_targets:
            return {"error": "No investment targets for allocation"}

        # 计算总权重并标准化
        total_weight = sum(target["allocation_weight"] for target in investment_targets)

        # 标准化权重
        for target in investment_targets:
            target["normalized_weight"] = round(target["allocation_weight"] / total_weight, 3)

        # 生成配置建议
        allocation_strategy = {
            "strategy_type": "Strategic Asset Allocation",
            "rebalancing_frequency": "quarterly",
            "investment_horizon": "12-18 months",
            "total_positions": len(investment_targets),
            "sector_breakdown": {},
            "position_sizing": {
                "max_single_position": 0.15,  # 单一持仓不超过15%
                "min_position_size": 0.02,    # 最小持仓2%
                "cash_reserve": 0.05          # 保留5%现金
            },
            "implementation_approach": {
                "entry_strategy": "分批建仓，3-4周内完成",
                "monitoring_frequency": "weekly",
                "rebalancing_threshold": "权重偏离超过2%时调整",
                "exit_criteria": "基本面恶化或技术面破位"
            },
            "performance_targets": {
                "expected_annual_return": round(sum(target["expected_return"] * target["normalized_weight"]
                                                  for target in investment_targets), 2),
                "target_sharpe_ratio": 1.2,
                "max_drawdown_tolerance": 0.15
            }
        }

        # 按板块汇总
        for target in investment_targets:
            sector = target["sector"]
            if sector not in allocation_strategy["sector_breakdown"]:
                allocation_strategy["sector_breakdown"][sector] = {
                    "weight": 0,
                    "positions": []
                }

            allocation_strategy["sector_breakdown"][sector]["weight"] += target["normalized_weight"]
            allocation_strategy["sector_breakdown"][sector]["positions"].append({
                "symbol": target["symbol"],
                "weight": target["normalized_weight"],
                "expected_return": target["expected_return"]
            })

        # 四舍五入权重
        for sector_data in allocation_strategy["sector_breakdown"].values():
            sector_data["weight"] = round(sector_data["weight"], 3)

        return allocation_strategy

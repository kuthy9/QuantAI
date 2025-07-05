"""
Performance Attribution Agent for QuantAI.

This agent provides comprehensive performance attribution analysis including:
- Strategy-level performance decomposition
- Factor-based attribution analysis
- Sector and asset class attribution
- Risk-adjusted performance metrics
- Rolling attribution analysis
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from loguru import logger

from ...core.base import BaseAgent, AgentMessage, MessageType
from ...analytics.performance_attribution import (
    PerformanceAttributionEngine,
    AttributionType,
    PerformanceMetric
)


class PerformanceAttributionAgent(BaseAgent):
    """
    Performance Attribution Agent.
    
    Provides comprehensive performance attribution analysis across multiple dimensions
    including strategies, factors, sectors, and individual securities.
    """
    
    def __init__(self, agent_id: str = "performance_attribution_agent", **kwargs):
        """Initialize the Performance Attribution Agent."""
        super().__init__(agent_id=agent_id, **kwargs)
        
        # Initialize attribution engine
        attribution_config = self.config.get('attribution', {})
        self.attribution_engine = PerformanceAttributionEngine(attribution_config)
        
        # Attribution parameters
        self.default_periods = self.config.get('default_periods', [30, 90, 252])
        self.auto_update_interval = self.config.get('auto_update_interval', 3600)  # 1 hour
        self.min_observations = self.config.get('min_observations', 20)
        
        # Data cache
        self.last_update = None
        self.cached_reports = {}
        
        logger.info(f"Performance Attribution Agent {agent_id} initialized")
    
    async def handle_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming messages for performance attribution analysis."""
        
        try:
            if message.message_type == MessageType.PERFORMANCE_ATTRIBUTION_REQUEST:
                return await self._handle_attribution_request(message)
            
            elif message.message_type == MessageType.PERFORMANCE_SUMMARY_REQUEST:
                return await self._handle_summary_request(message)
            
            elif message.message_type == MessageType.ROLLING_ATTRIBUTION_REQUEST:
                return await self._handle_rolling_attribution_request(message)
            
            elif message.message_type == MessageType.FACTOR_ATTRIBUTION_REQUEST:
                return await self._handle_factor_attribution_request(message)
            
            elif message.message_type == MessageType.SECTOR_ATTRIBUTION_REQUEST:
                return await self._handle_sector_attribution_request(message)
            
            elif message.message_type == MessageType.RETURNS_DATA_UPDATE:
                return await self._handle_returns_data_update(message)
            
            else:
                logger.warning(f"Unknown message type: {message.message_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.ERROR,
                content={
                    'error': str(e),
                    'original_request': message.content
                }
            )
    
    async def _handle_attribution_request(self, message: AgentMessage) -> AgentMessage:
        """Handle comprehensive attribution analysis request."""
        
        request_data = message.content
        strategy_ids = request_data.get('strategy_ids', [])
        period_days = request_data.get('period_days', 252)
        include_factor_analysis = request_data.get('include_factor_analysis', True)
        account_id = request_data.get('account_id')
        
        try:
            # Generate comprehensive attribution report
            attribution_report = await self.attribution_engine.generate_attribution_report(
                strategy_ids=strategy_ids,
                period_days=period_days,
                include_factor_analysis=include_factor_analysis
            )
            
            # Cache the report
            cache_key = f"{account_id}_{period_days}_{len(strategy_ids)}"
            self.cached_reports[cache_key] = {
                'report': attribution_report,
                'timestamp': datetime.now()
            }
            
            # Save to file if requested
            if request_data.get('save_report', False):
                filename = request_data.get('filename')
                await self.attribution_engine.save_attribution_results(
                    attribution_report, filename
                )
            
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.PERFORMANCE_ATTRIBUTION_RESPONSE,
                content={
                    'attribution_report': attribution_report,
                    'request_id': request_data.get('request_id'),
                    'status': 'success'
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating attribution report: {e}")
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.ERROR,
                content={
                    'error': f"Attribution analysis failed: {str(e)}",
                    'request_id': request_data.get('request_id')
                }
            )
    
    async def _handle_summary_request(self, message: AgentMessage) -> AgentMessage:
        """Handle performance summary request."""
        
        request_data = message.content
        account_id = request_data.get('account_id')
        period_days = request_data.get('period_days', 30)
        
        try:
            # Get portfolio performance summary
            summary = await self.attribution_engine.get_portfolio_performance_summary(
                account_id=account_id,
                period_days=period_days
            )
            
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.PERFORMANCE_SUMMARY_RESPONSE,
                content={
                    'performance_summary': summary,
                    'request_id': request_data.get('request_id'),
                    'status': 'success'
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.ERROR,
                content={
                    'error': f"Performance summary failed: {str(e)}",
                    'request_id': request_data.get('request_id')
                }
            )
    
    async def _handle_rolling_attribution_request(self, message: AgentMessage) -> AgentMessage:
        """Handle rolling attribution analysis request."""
        
        request_data = message.content
        strategy_id = request_data.get('strategy_id')
        window_days = request_data.get('window_days', 30)
        step_days = request_data.get('step_days', 7)
        
        if not strategy_id:
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.ERROR,
                content={
                    'error': "strategy_id is required for rolling attribution",
                    'request_id': request_data.get('request_id')
                }
            )
        
        try:
            # Calculate rolling attribution
            rolling_results = await self.attribution_engine.calculate_rolling_attribution(
                strategy_id=strategy_id,
                window_days=window_days,
                step_days=step_days
            )
            
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.ROLLING_ATTRIBUTION_RESPONSE,
                content={
                    'rolling_attribution': rolling_results,
                    'strategy_id': strategy_id,
                    'window_days': window_days,
                    'step_days': step_days,
                    'request_id': request_data.get('request_id'),
                    'status': 'success'
                }
            )
            
        except Exception as e:
            logger.error(f"Error calculating rolling attribution: {e}")
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.ERROR,
                content={
                    'error': f"Rolling attribution failed: {str(e)}",
                    'request_id': request_data.get('request_id')
                }
            )
    
    async def _handle_factor_attribution_request(self, message: AgentMessage) -> AgentMessage:
        """Handle factor attribution analysis request."""
        
        request_data = message.content
        strategy_id = request_data.get('strategy_id')
        factor_model = request_data.get('factor_model', 'fama_french_3')
        period_days = request_data.get('period_days', 252)
        
        if not strategy_id:
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.ERROR,
                content={
                    'error': "strategy_id is required for factor attribution",
                    'request_id': request_data.get('request_id')
                }
            )
        
        try:
            # Calculate factor attribution
            factor_attribution = await self.attribution_engine.calculate_factor_attribution(
                strategy_id=strategy_id,
                factor_model=factor_model,
                period_days=period_days
            )
            
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.FACTOR_ATTRIBUTION_RESPONSE,
                content={
                    'factor_attribution': factor_attribution,
                    'strategy_id': strategy_id,
                    'factor_model': factor_model,
                    'request_id': request_data.get('request_id'),
                    'status': 'success'
                }
            )
            
        except Exception as e:
            logger.error(f"Error calculating factor attribution: {e}")
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.ERROR,
                content={
                    'error': f"Factor attribution failed: {str(e)}",
                    'request_id': request_data.get('request_id')
                }
            )
    
    async def _handle_sector_attribution_request(self, message: AgentMessage) -> AgentMessage:
        """Handle sector attribution analysis request."""
        
        request_data = message.content
        portfolio_positions = request_data.get('portfolio_positions', {})
        benchmark_weights = request_data.get('benchmark_weights', {})
        period_days = request_data.get('period_days', 30)
        
        if not portfolio_positions or not benchmark_weights:
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.ERROR,
                content={
                    'error': "portfolio_positions and benchmark_weights are required",
                    'request_id': request_data.get('request_id')
                }
            )
        
        try:
            # Calculate sector attribution
            sector_attribution = await self.attribution_engine.calculate_sector_attribution(
                portfolio_positions=portfolio_positions,
                benchmark_weights=benchmark_weights,
                period_days=period_days
            )
            
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.SECTOR_ATTRIBUTION_RESPONSE,
                content={
                    'sector_attribution': sector_attribution,
                    'period_days': period_days,
                    'request_id': request_data.get('request_id'),
                    'status': 'success'
                }
            )
            
        except Exception as e:
            logger.error(f"Error calculating sector attribution: {e}")
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.ERROR,
                content={
                    'error': f"Sector attribution failed: {str(e)}",
                    'request_id': request_data.get('request_id')
                }
            )
    
    async def _handle_returns_data_update(self, message: AgentMessage) -> AgentMessage:
        """Handle returns data update."""
        
        request_data = message.content
        returns_data = request_data.get('returns_data', {})
        benchmark_returns = request_data.get('benchmark_returns')
        
        try:
            # Convert data to pandas Series if needed
            processed_returns = {}
            for strategy_id, data in returns_data.items():
                if isinstance(data, dict):
                    # Convert dict to pandas Series
                    processed_returns[strategy_id] = pd.Series(data)
                elif isinstance(data, pd.Series):
                    processed_returns[strategy_id] = data
                else:
                    logger.warning(f"Unknown data format for strategy {strategy_id}")
            
            # Process benchmark data
            processed_benchmark = None
            if benchmark_returns:
                if isinstance(benchmark_returns, dict):
                    processed_benchmark = pd.Series(benchmark_returns)
                elif isinstance(benchmark_returns, pd.Series):
                    processed_benchmark = benchmark_returns
            
            # Load data into attribution engine
            await self.attribution_engine.load_returns_data(
                processed_returns, processed_benchmark
            )
            
            self.last_update = datetime.now()
            
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.DATA_UPDATE_CONFIRMATION,
                content={
                    'status': 'success',
                    'strategies_loaded': len(processed_returns),
                    'benchmark_loaded': processed_benchmark is not None,
                    'update_timestamp': self.last_update.isoformat(),
                    'request_id': request_data.get('request_id')
                }
            )
            
        except Exception as e:
            logger.error(f"Error updating returns data: {e}")
            return AgentMessage(
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                message_type=MessageType.ERROR,
                content={
                    'error': f"Returns data update failed: {str(e)}",
                    'request_id': request_data.get('request_id')
                }
            )
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics."""
        
        return {
            'agent_id': self.agent_id,
            'status': 'active',
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'cached_reports': len(self.cached_reports),
            'strategies_loaded': len(self.attribution_engine.returns_data),
            'benchmark_loaded': self.attribution_engine.benchmark_data is not None,
            'default_periods': self.default_periods,
            'auto_update_interval': self.auto_update_interval,
            'min_observations': self.min_observations
        }
    
    async def cleanup_cache(self, max_age_hours: int = 24):
        """Clean up old cached reports."""
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        keys_to_remove = []
        for key, cached_data in self.cached_reports.items():
            if cached_data['timestamp'] < cutoff_time:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.cached_reports[key]
        
        logger.info(f"Cleaned up {len(keys_to_remove)} cached reports")
    
    async def start(self):
        """Start the Performance Attribution Agent."""
        await super().start()
        
        # Start periodic cache cleanup
        asyncio.create_task(self._periodic_cache_cleanup())
        
        logger.info(f"Performance Attribution Agent {self.agent_id} started")
    
    async def _periodic_cache_cleanup(self):
        """Periodic cache cleanup task."""
        
        while self.is_running:
            try:
                await self.cleanup_cache()
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Error in periodic cache cleanup: {e}")
                await asyncio.sleep(3600)

"""
Backtest Monitor Agent for the QuantAI multi-agent system.

This agent is responsible for monitoring backtesting processes
and analyzing strategy performance in historical data.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from autogen_core import Agent, MessageContext
from ...core.base import BaseQuantAgent, AgentRole, AgentCapability
from ...core.messages import QuantMessage, MessageType, BacktestMessage


class BacktestMonitorAgent(BaseQuantAgent):
    """
    Agent responsible for monitoring backtesting processes.
    
    This agent manages backtesting workflows, monitors performance,
    and provides analysis of strategy effectiveness on historical data.
    """
    
    def __init__(self, agent_id: str = "backtest_monitor"):
        super().__init__(
            role=AgentRole.BACKTEST_MONITOR,
            capabilities=[
                AgentCapability.PERFORMANCE_MONITORING,
                AgentCapability.DATA_ANALYSIS,
                AgentCapability.BACKTESTING,
            ]
        )
        self.agent_id = agent_id
        self.active_backtests: Dict[str, Dict[str, Any]] = {}
        self.completed_backtests: List[Dict[str, Any]] = []
        self.performance_metrics = {
            'total_backtests': 0,
            'successful_backtests': 0,
            'failed_backtests': 0,
            'average_duration': 0.0
        }
    
    async def on_messages(self, messages: List[QuantMessage], ctx: MessageContext) -> str:
        """Handle incoming messages for backtest monitoring."""
        results = []
        
        for message in messages:
            if isinstance(message, BacktestMessage):
                result = await self._handle_backtest_request(message)
                results.append(result)
            else:
                result = await self._handle_general_message(message)
                results.append(result)
        
        return f"BacktestMonitorAgent processed {len(results)} messages"
    
    async def _handle_backtest_request(self, message: BacktestMessage) -> Dict[str, Any]:
        """Handle backtest request."""
        try:
            backtest_id = message.backtest_id
            
            # Validate backtest request
            if not await self._validate_backtest_request(message):
                return {
                    'status': 'rejected',
                    'backtest_id': backtest_id,
                    'error': 'Backtest validation failed'
                }
            
            # Start backtest
            backtest_result = await self._run_backtest(message)
            
            # Record backtest
            self.completed_backtests.append({
                'backtest_id': backtest_id,
                'completion_time': datetime.now(),
                'status': backtest_result['status'],
                'results': backtest_result
            })
            
            # Update metrics
            self._update_backtest_metrics(backtest_result)
            
            return backtest_result
            
        except Exception as e:
            return {
                'status': 'error',
                'backtest_id': getattr(message, 'backtest_id', 'unknown'),
                'error': str(e)
            }
    
    async def _validate_backtest_request(self, backtest: BacktestMessage) -> bool:
        """Validate backtest request."""
        # Check required fields
        if not backtest.strategy_id:
            return False
        
        if not backtest.start_date or not backtest.end_date:
            return False
        
        if backtest.start_date >= backtest.end_date:
            return False
        
        if not backtest.symbols:
            return False
        
        if backtest.initial_capital <= 0:
            return False
        
        return True
    
    async def _run_backtest(self, backtest: BacktestMessage) -> Dict[str, Any]:
        """Run backtest simulation."""
        backtest_id = backtest.backtest_id
        
        # Add to active backtests
        self.active_backtests[backtest_id] = {
            'strategy_id': backtest.strategy_id,
            'start_date': backtest.start_date,
            'end_date': backtest.end_date,
            'symbols': backtest.symbols,
            'initial_capital': backtest.initial_capital,
            'start_time': datetime.now(),
            'status': 'running'
        }
        
        # Simulate backtest execution
        await asyncio.sleep(0.5)  # Simulate backtest time
        
        # Generate simulated results
        results = await self._generate_backtest_results(backtest)
        
        # Remove from active backtests
        if backtest_id in self.active_backtests:
            del self.active_backtests[backtest_id]
        
        return {
            'status': 'completed',
            'backtest_id': backtest_id,
            'strategy_id': backtest.strategy_id,
            'execution_time': datetime.now(),
            'results': results,
            'message': f'Backtest {backtest_id} completed successfully'
        }
    
    async def _generate_backtest_results(self, backtest: BacktestMessage) -> Dict[str, Any]:
        """Generate simulated backtest results."""
        import random
        
        # Calculate backtest period
        period_days = (backtest.end_date - backtest.start_date).days
        
        # Simulate performance metrics
        annual_return = random.uniform(-0.2, 0.4)  # -20% to 40% annual return
        volatility = random.uniform(0.1, 0.3)  # 10% to 30% volatility
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        max_drawdown = random.uniform(0.05, 0.25)  # 5% to 25% max drawdown
        
        # Simulate trade statistics
        total_trades = random.randint(50, 500)
        winning_trades = random.randint(int(total_trades * 0.3), int(total_trades * 0.7))
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate final portfolio value
        final_value = backtest.initial_capital * (1 + annual_return * period_days / 365)
        
        return {
            'performance_metrics': {
                'total_return': (final_value - backtest.initial_capital) / backtest.initial_capital,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'final_value': final_value
            },
            'trade_statistics': {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'win_rate': win_rate,
                'average_trade_return': annual_return / total_trades if total_trades > 0 else 0
            },
            'period_info': {
                'start_date': backtest.start_date,
                'end_date': backtest.end_date,
                'period_days': period_days,
                'symbols': backtest.symbols,
                'initial_capital': backtest.initial_capital
            }
        }
    
    async def _handle_general_message(self, message: QuantMessage) -> Dict[str, Any]:
        """Handle general messages."""
        return {
            'status': 'processed',
            'message_type': message.message_type.value,
            'sender': message.sender_id
        }

    async def process_message(self, message: QuantMessage, ctx: MessageContext) -> Optional[QuantMessage]:
        """Process a single message (required by BaseQuantAgent)."""
        try:
            if isinstance(message, BacktestMessage):
                result = await self._handle_backtest_request(message)
                return QuantMessage(
                    message_type=MessageType.BACKTEST_RESPONSE,
                    sender_id=self.agent_id,
                    data_payload=result
                )
            else:
                result = await self._handle_general_message(message)
                return QuantMessage(
                    message_type=MessageType.GENERAL_RESPONSE,
                    sender_id=self.agent_id,
                    data_payload=result
                )
        except Exception as e:
            return QuantMessage(
                message_type=MessageType.ERROR,
                sender_id=self.agent_id,
                error_message=str(e)
            )
    
    def _update_backtest_metrics(self, backtest_result: Dict[str, Any]) -> None:
        """Update backtest metrics."""
        self.performance_metrics['total_backtests'] += 1
        
        if backtest_result['status'] == 'completed':
            self.performance_metrics['successful_backtests'] += 1
        else:
            self.performance_metrics['failed_backtests'] += 1
    
    def get_active_backtests(self) -> Dict[str, Dict[str, Any]]:
        """Get list of active backtests."""
        return self.active_backtests.copy()
    
    def get_backtest_history(self) -> List[Dict[str, Any]]:
        """Get backtest history."""
        return self.completed_backtests.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        return self.performance_metrics.copy()

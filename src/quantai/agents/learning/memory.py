"""
Memory Agent (D6) for the QuantAI system.

This agent stores strategy results and insights in long-term memory,
providing context and historical knowledge for future decisions.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from autogen_core import MessageContext
from autogen_core.models import ChatCompletionClient, UserMessage
from loguru import logger

from ...core.base import AgentCapability, AgentRole, ModelCapableAgent
from ...core.memory import QuantMemory, StrategyMemory, MarketMemory, PerformanceMemory
from ...core.messages import (
    MemoryMessage, 
    MessageType, 
    QuantMessage, 
    StrategyMessage,
    BacktestMessage,
    FeedbackMessage
)


class MemoryAgent(ModelCapableAgent):
    """
    Memory Agent (D6) - Manages long-term memory and institutional knowledge.
    
    Capabilities:
    - Strategy performance and outcome storage
    - Market regime and pattern memory
    - Cross-strategy learning and insights
    - Historical context retrieval
    - Knowledge graph construction
    - Institutional memory preservation
    """
    
    def __init__(
        self,
        model_client: ChatCompletionClient,
        memory_retention_days: int = 365 * 3,  # 3 years
        similarity_threshold: float = 0.7,
        max_memory_items: int = 10000,
        **kwargs
    ):
        super().__init__(
            role=AgentRole.MEMORY,
            capabilities=[
                AgentCapability.MEMORY_MANAGEMENT,
                AgentCapability.LEARNING,
            ],
            model_client=model_client,
            system_message=self._get_system_message(),
            **kwargs
        )
        
        self.memory_retention_days = memory_retention_days
        self.similarity_threshold = similarity_threshold
        self.max_memory_items = max_memory_items
        
        # Initialize memory system
        from ...core.config import get_config
        config = get_config()
        self.memory = QuantMemory(config.memory)
        
        # Memory categories and indexing
        self._memory_categories = {
            "strategy_outcomes": "Strategy performance and results",
            "market_regimes": "Market conditions and regime changes", 
            "learning_insights": "Extracted learnings and patterns",
            "failure_analysis": "Failed strategy analysis and causes",
            "success_patterns": "Successful strategy characteristics",
            "risk_events": "Risk management events and responses",
            "market_events": "Significant market events and impacts",
        }
        
        # Knowledge graph for relationships
        self._knowledge_graph: Dict[str, Dict[str, Any]] = {}
        self._entity_relationships: Dict[str, List[str]] = {}
        
        # Memory statistics
        self._memory_stats = {
            "total_entries": 0,
            "categories": {cat: 0 for cat in self._memory_categories},
            "last_cleanup": None,
            "retrieval_count": 0,
            "storage_count": 0,
        }
    
    def _get_system_message(self) -> str:
        return """You are a Memory Agent responsible for maintaining institutional knowledge and long-term memory for the trading system.

Your responsibilities:
1. Store and organize strategy outcomes, market insights, and learning data
2. Maintain historical context and institutional knowledge
3. Provide relevant historical information for decision making
4. Build knowledge graphs connecting related concepts and patterns
5. Preserve critical learnings and prevent knowledge loss
6. Enable cross-temporal learning and pattern recognition

Memory Management Framework:

1. Strategy Memory
   - Performance outcomes and metrics
   - Implementation details and code
   - Market conditions during execution
   - Success and failure factors
   - Lessons learned and insights

2. Market Memory
   - Regime changes and transitions
   - Significant market events
   - Economic cycle patterns
   - Volatility and correlation changes
   - Cross-asset relationships

3. Learning Memory
   - Pattern recognition insights
   - Systematic biases identified
   - Process improvements discovered
   - Risk management learnings
   - Adaptation strategies

4. Contextual Memory
   - Decision rationales and outcomes
   - Agent interactions and coordination
   - System evolution and changes
   - External factor impacts
   - Regulatory and compliance events

Knowledge Organization:

1. Temporal Indexing
   - Time-based organization of events
   - Regime and cycle alignment
   - Seasonal and calendar effects
   - Evolution tracking over time

2. Conceptual Clustering
   - Similar strategy groupings
   - Market condition categories
   - Performance pattern clusters
   - Risk factor associations

3. Relationship Mapping
   - Cause-effect relationships
   - Strategy correlations
   - Market factor dependencies
   - Cross-domain connections

4. Importance Weighting
   - Significance scoring of memories
   - Relevance to current conditions
   - Learning value assessment
   - Decision impact measurement

Retrieval and Application:

1. Context-Aware Retrieval
   - Current market regime matching
   - Similar strategy identification
   - Analogous situation finding
   - Pattern completion assistance

2. Proactive Insights
   - Early warning pattern recognition
   - Opportunity identification
   - Risk factor monitoring
   - Adaptation recommendations

3. Knowledge Transfer
   - Cross-strategy learning application
   - Historical precedent identification
   - Best practice propagation
   - Failure prevention guidance

Guidelines:
- Maintain comprehensive but organized memory
- Prioritize actionable and relevant information
- Build rich contextual relationships
- Enable efficient retrieval and application
- Preserve critical institutional knowledge
- Support continuous learning and improvement

Focus on creating a living memory system that enhances decision-making through historical wisdom and pattern recognition."""
    
    async def process_message(
        self, 
        message: QuantMessage, 
        ctx: MessageContext
    ) -> Optional[QuantMessage]:
        """Process memory storage and retrieval requests."""
        
        if isinstance(message, MemoryMessage):
            if message.operation == "store":
                await self._store_memory(message)
            elif message.operation == "retrieve":
                results = await self._retrieve_memory(message)
                
                # Create response with retrieved memories
                response = MemoryMessage(
                    message_type=MessageType.MEMORY_UPDATE,
                    sender_id=self.agent_id,
                    memory_type=message.memory_type,
                    operation="retrieve_response",
                    key=message.key,
                    query=message.query,
                    results=results,
                    session_id=message.session_id,
                    correlation_id=message.correlation_id,
                )
                
                self._memory_stats["retrieval_count"] += 1
                return response
        
        # Auto-store important messages
        elif isinstance(message, (StrategyMessage, BacktestMessage, FeedbackMessage)):
            await self._auto_store_message(message)
        
        return None
    
    async def _store_memory(self, memory_message: MemoryMessage):
        """Store a memory entry."""
        
        memory_type = memory_message.memory_type
        key = memory_message.key
        value = memory_message.value
        
        logger.info(f"Storing memory: {memory_type} - {key}")
        
        try:
            if memory_type == "strategy":
                await self._store_strategy_memory(key, value)
            elif memory_type == "market":
                await self._store_market_memory(key, value)
            elif memory_type == "performance":
                await self._store_performance_memory(key, value)
            elif memory_type == "learning":
                await self._store_learning_memory(key, value)
            else:
                # Generic memory storage
                await self._store_generic_memory(memory_type, key, value)
            
            # Update statistics
            self._memory_stats["storage_count"] += 1
            self._memory_stats["total_entries"] += 1
            if memory_type in self._memory_stats["categories"]:
                self._memory_stats["categories"][memory_type] += 1
            
            # Update knowledge graph
            await self._update_knowledge_graph(memory_type, key, value)
            
        except Exception as e:
            logger.error(f"Error storing memory {key}: {e}")
    
    async def _store_strategy_memory(self, key: str, value: Dict[str, Any]):
        """Store strategy-related memory."""
        
        strategy_data = value
        
        memory_id = await self.memory.store_strategy(
            strategy_id=strategy_data.get("strategy_id", key),
            strategy_name=strategy_data.get("name", "Unknown Strategy"),
            strategy_code=strategy_data.get("code", ""),
            performance_metrics=strategy_data.get("performance_metrics", {}),
            market_conditions=strategy_data.get("market_conditions", {}),
            metadata={
                "stored_by": self.agent_id,
                "category": "strategy_outcomes",
                "importance": strategy_data.get("importance", 0.5),
            }
        )
        
        logger.debug(f"Stored strategy memory with ID: {memory_id}")
    
    async def _store_market_memory(self, key: str, value: Dict[str, Any]):
        """Store market-related memory."""
        
        market_data = value
        
        memory_id = await self.memory.store_market_insight(
            symbol=market_data.get("symbol", "MARKET"),
            timeframe=market_data.get("timeframe", "daily"),
            market_regime=market_data.get("regime", "unknown"),
            insights=market_data.get("insights", []),
            technical_indicators=market_data.get("technical_indicators", {}),
            sentiment_score=market_data.get("sentiment_score"),
            metadata={
                "stored_by": self.agent_id,
                "category": "market_regimes",
                "importance": market_data.get("importance", 0.5),
            }
        )
        
        logger.debug(f"Stored market memory with ID: {memory_id}")
    
    async def _store_performance_memory(self, key: str, value: Dict[str, Any]):
        """Store performance-related memory."""
        
        perf_data = value
        
        memory_id = await self.memory.store_performance(
            strategy_id=perf_data.get("strategy_id", key),
            period_start=perf_data.get("period_start", datetime.now() - timedelta(days=30)),
            period_end=perf_data.get("period_end", datetime.now()),
            returns=perf_data.get("returns", 0.0),
            sharpe_ratio=perf_data.get("sharpe_ratio", 0.0),
            max_drawdown=perf_data.get("max_drawdown", 0.0),
            win_rate=perf_data.get("win_rate", 0.0),
            lessons_learned=perf_data.get("lessons_learned", []),
            metadata={
                "stored_by": self.agent_id,
                "category": "strategy_outcomes",
                "importance": perf_data.get("importance", 0.5),
            }
        )
        
        logger.debug(f"Stored performance memory with ID: {memory_id}")
    
    async def _store_learning_memory(self, key: str, value: Dict[str, Any]):
        """Store learning insights and patterns."""
        
        learning_data = value
        
        # Create a comprehensive learning insight entry
        content = f"Learning Insight: {key}\n\n"
        content += f"Insights: {json.dumps(learning_data.get('insights', []), indent=2)}\n"
        content += f"Patterns: {json.dumps(learning_data.get('patterns', {}), indent=2)}\n"
        content += f"Recommendations: {json.dumps(learning_data.get('recommendations', []), indent=2)}"
        
        from ...core.memory import MemoryEntry
        
        entry = MemoryEntry(
            content=content,
            metadata={
                "type": "learning_insight",
                "category": "learning_insights",
                "stored_by": self.agent_id,
                "importance": learning_data.get("importance", 0.7),
                "insights": learning_data.get("insights", []),
                "patterns": learning_data.get("patterns", {}),
                "recommendations": learning_data.get("recommendations", []),
            }
        )
        
        memory_id = await self.memory.store.store(entry)
        logger.debug(f"Stored learning memory with ID: {memory_id}")
    
    async def _store_generic_memory(self, memory_type: str, key: str, value: Any):
        """Store generic memory entry."""
        
        from ...core.memory import MemoryEntry
        
        content = f"{memory_type}: {key}\n\n{json.dumps(value, indent=2, default=str)}"
        
        entry = MemoryEntry(
            content=content,
            metadata={
                "type": memory_type,
                "category": memory_type,
                "stored_by": self.agent_id,
                "key": key,
                "importance": 0.5,
            }
        )
        
        memory_id = await self.memory.store.store(entry)
        logger.debug(f"Stored generic memory with ID: {memory_id}")
    
    async def _retrieve_memory(self, memory_message: MemoryMessage) -> List[Dict[str, Any]]:
        """Retrieve memory entries based on query."""
        
        query = memory_message.query
        memory_type = memory_message.memory_type
        
        logger.info(f"Retrieving memory: {memory_type} - {query}")
        
        try:
            if memory_type == "strategy":
                results = await self.memory.search_strategies(query, limit=10)
            elif memory_type == "market":
                results = await self.memory.search_market_insights(query, limit=10)
            else:
                # Generic search
                results = await self.memory.store.search(
                    query=query,
                    limit=10,
                    filters={"type": memory_type} if memory_type else None
                )
            
            # Convert results to dictionaries
            memory_results = []
            for result in results:
                memory_results.append({
                    "id": result.id,
                    "content": result.content,
                    "metadata": result.metadata,
                    "created_at": result.created_at.isoformat(),
                    "relevance_score": getattr(result, "relevance_score", 0.0),
                })
            
            # Enhance with contextual information
            enhanced_results = await self._enhance_retrieval_results(memory_results, query)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error retrieving memory for query '{query}': {e}")
            return []
    
    async def _enhance_retrieval_results(
        self, 
        results: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """Enhance retrieval results with additional context."""
        
        enhanced_results = []
        
        for result in results:
            enhanced_result = result.copy()
            
            # Add related memories
            related_memories = await self._find_related_memories(result["id"], limit=3)
            enhanced_result["related_memories"] = related_memories
            
            # Add temporal context
            temporal_context = await self._get_temporal_context(result)
            enhanced_result["temporal_context"] = temporal_context
            
            # Add relevance explanation
            relevance_explanation = await self._explain_relevance(result, query)
            enhanced_result["relevance_explanation"] = relevance_explanation
            
            enhanced_results.append(enhanced_result)
        
        return enhanced_results
    
    async def _find_related_memories(self, memory_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Find memories related to a given memory."""
        
        # This would use the knowledge graph to find related memories
        # For now, return empty list
        return []
    
    async def _get_temporal_context(self, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Get temporal context for a memory."""
        
        created_at = datetime.fromisoformat(memory["created_at"])
        now = datetime.now()
        
        age_days = (now - created_at).days
        
        # Determine temporal relevance
        if age_days < 30:
            temporal_relevance = "recent"
        elif age_days < 90:
            temporal_relevance = "medium_term"
        elif age_days < 365:
            temporal_relevance = "long_term"
        else:
            temporal_relevance = "historical"
        
        return {
            "age_days": age_days,
            "temporal_relevance": temporal_relevance,
            "created_at": memory["created_at"],
        }
    
    async def _explain_relevance(self, memory: Dict[str, Any], query: str) -> str:
        """Explain why a memory is relevant to the query."""
        
        # Use LLM to explain relevance
        prompt = f"""Explain why this memory is relevant to the query:

Query: {query}

Memory Content: {memory['content'][:500]}...

Memory Metadata: {json.dumps(memory['metadata'], indent=2)}

Provide a brief explanation of the relevance:"""
        
        try:
            response = await self._call_model([UserMessage(content=prompt, source="user")])
            return response.strip()
        except Exception as e:
            logger.error(f"Error explaining relevance: {e}")
            return "Relevance based on content similarity"
    
    async def _auto_store_message(self, message: QuantMessage):
        """Automatically store important messages in memory."""
        
        try:
            if isinstance(message, StrategyMessage):
                await self._auto_store_strategy_message(message)
            elif isinstance(message, BacktestMessage):
                await self._auto_store_backtest_message(message)
            elif isinstance(message, FeedbackMessage):
                await self._auto_store_feedback_message(message)
                
        except Exception as e:
            logger.error(f"Error auto-storing message: {e}")
    
    async def _auto_store_strategy_message(self, message: StrategyMessage):
        """Auto-store strategy message."""
        
        if message.message_type == MessageType.STRATEGY_CODE:
            # Store coded strategy
            strategy_data = {
                "strategy_id": message.strategy_id,
                "name": message.strategy_name,
                "description": message.strategy_description,
                "code": message.strategy_code,
                "parameters": message.strategy_parameters,
                "timestamp": datetime.now(),
                "importance": 0.6,
            }
            
            await self._store_strategy_memory(message.strategy_id, strategy_data)
    
    async def _auto_store_backtest_message(self, message: BacktestMessage):
        """Auto-store backtest results."""
        
        if message.message_type == MessageType.BACKTEST_RESULT:
            # Store backtest performance
            perf_data = {
                "strategy_id": message.strategy_id,
                "period_start": message.start_date,
                "period_end": message.end_date,
                "returns": message.performance_metrics.get("annual_return", 0.0),
                "sharpe_ratio": message.performance_metrics.get("sharpe_ratio", 0.0),
                "max_drawdown": message.performance_metrics.get("max_drawdown", 0.0),
                "win_rate": message.performance_metrics.get("win_rate", 0.0),
                "importance": 0.7,
            }
            
            await self._store_performance_memory(message.strategy_id, perf_data)
    
    async def _auto_store_feedback_message(self, message: FeedbackMessage):
        """Auto-store feedback and learning insights."""
        
        learning_data = {
            "strategy_id": message.strategy_id,
            "insights": message.lessons_learned,
            "recommendations": message.improvement_suggestions,
            "success_factors": message.success_factors,
            "failure_factors": message.failure_factors,
            "performance_gap": {
                metric: actual - message.performance_expected.get(metric, 0)
                for metric, actual in message.performance_actual.items()
            },
            "importance": 0.8,
        }
        
        await self._store_learning_memory(f"feedback_{message.strategy_id}", learning_data)
    
    async def _update_knowledge_graph(self, memory_type: str, key: str, value: Any):
        """Update the knowledge graph with new memory relationships."""
        
        # Create or update entity in knowledge graph
        entity_id = f"{memory_type}:{key}"
        
        if entity_id not in self._knowledge_graph:
            self._knowledge_graph[entity_id] = {
                "type": memory_type,
                "key": key,
                "created_at": datetime.now(),
                "connections": [],
                "importance": 0.5,
            }
        
        # Find and create relationships
        await self._identify_relationships(entity_id, value)
    
    async def _identify_relationships(self, entity_id: str, value: Any):
        """Identify relationships between entities."""
        
        # This would implement sophisticated relationship detection
        # For now, implement basic relationship detection
        
        if isinstance(value, dict):
            # Look for strategy_id references
            if "strategy_id" in value:
                strategy_entity = f"strategy:{value['strategy_id']}"
                self._add_relationship(entity_id, strategy_entity, "references")
            
            # Look for market regime references
            if "market_regime" in value:
                regime_entity = f"market_regime:{value['market_regime']}"
                self._add_relationship(entity_id, regime_entity, "occurs_in")
    
    def _add_relationship(self, entity1: str, entity2: str, relationship_type: str):
        """Add a relationship between two entities."""
        
        if entity1 in self._knowledge_graph:
            self._knowledge_graph[entity1]["connections"].append({
                "target": entity2,
                "type": relationship_type,
                "created_at": datetime.now(),
            })
        
        if entity2 not in self._entity_relationships:
            self._entity_relationships[entity2] = []
        
        self._entity_relationships[entity2].append(entity1)
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        
        await self.memory.initialize()
        
        # Get recent memories
        recent_memories = await self.memory.get_recent_insights(limit=50)
        
        return {
            "total_entries": self._memory_stats["total_entries"],
            "categories": self._memory_stats["categories"],
            "retrieval_count": self._memory_stats["retrieval_count"],
            "storage_count": self._memory_stats["storage_count"],
            "recent_entries": len(recent_memories),
            "knowledge_graph_entities": len(self._knowledge_graph),
            "last_cleanup": self._memory_stats["last_cleanup"],
            "memory_categories": list(self._memory_categories.keys()),
        }
    
    async def search_similar_strategies(
        self, 
        strategy_description: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for strategies similar to the given description."""
        
        results = await self.memory.search_strategies(strategy_description, limit=limit)
        
        return [
            {
                "strategy_id": result.strategy_id,
                "strategy_name": result.strategy_name,
                "performance_metrics": result.performance_metrics,
                "market_conditions": result.market_conditions,
                "similarity_score": getattr(result, "similarity_score", 0.0),
            }
            for result in results
        ]
    
    async def get_market_regime_history(self, regime: str) -> List[Dict[str, Any]]:
        """Get historical data for a specific market regime."""
        
        results = await self.memory.search_market_insights(
            f"market regime {regime}",
            limit=20
        )
        
        return [
            {
                "symbol": result.symbol,
                "timeframe": result.timeframe,
                "market_regime": result.market_regime,
                "key_insights": result.key_insights,
                "technical_indicators": result.technical_indicators,
                "created_at": result.created_at.isoformat(),
            }
            for result in results
        ]
    
    async def cleanup_old_memories(self):
        """Clean up old and less important memories."""
        
        cutoff_date = datetime.now() - timedelta(days=self.memory_retention_days)
        
        # This would implement memory cleanup logic
        # For now, just update the cleanup timestamp
        self._memory_stats["last_cleanup"] = datetime.now().isoformat()
        
        logger.info(f"Memory cleanup completed. Cutoff date: {cutoff_date}")
    
    async def export_knowledge_graph(self) -> Dict[str, Any]:
        """Export the knowledge graph for analysis."""
        
        return {
            "entities": self._knowledge_graph,
            "relationships": self._entity_relationships,
            "statistics": {
                "total_entities": len(self._knowledge_graph),
                "total_relationships": sum(
                    len(entity["connections"]) 
                    for entity in self._knowledge_graph.values()
                ),
                "entity_types": list(set(
                    entity["type"] 
                    for entity in self._knowledge_graph.values()
                )),
            }
        }

"""
Multimodal Fusion Agent for the QuantAI multi-agent system.

This agent is responsible for fusing data from multiple sources and modalities
to create comprehensive market insights and analysis.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from autogen_core import Agent, MessageContext
from ...core.base import BaseQuantAgent, AgentRole, AgentCapability
from ...core.messages import QuantMessage, MessageType, DataMessage


class MultimodalFusionAgent(BaseQuantAgent):
    """
    Agent responsible for multimodal data fusion and analysis.
    
    This agent combines data from various sources including:
    - Market data (price, volume, technical indicators)
    - News and sentiment data
    - Economic indicators
    - Social media sentiment
    - Alternative data sources
    """
    
    def __init__(self, agent_id: str = "multimodal_fusion"):
        super().__init__(
            role=AgentRole.MULTIMODAL_FUSION,
            capabilities=[
                AgentCapability.MULTIMODAL_ANALYSIS,
                AgentCapability.DATA_PROCESSING,
                AgentCapability.DATA_FUSION,
            ]
        )
        self.agent_id = agent_id
        self.data_sources: Dict[str, List[Dict[str, Any]]] = {
            'market': [],
            'news': [],
            'sentiment': [],
            'economic': [],
            'social': [],
            'alternative': []
        }
        self.fusion_results: List[Dict[str, Any]] = []
        self.fusion_algorithms = ['weighted_average', 'attention', 'ensemble']
    
    async def on_messages(self, messages: List[QuantMessage], ctx: MessageContext) -> str:
        """Handle incoming messages for multimodal fusion."""
        results = []
        
        for message in messages:
            if isinstance(message, DataMessage):
                result = await self._handle_data_fusion(message)
                results.append(result)
            else:
                result = await self._handle_general_message(message)
                results.append(result)
        
        return f"MultimodalFusionAgent processed {len(results)} messages"
    
    async def _handle_data_fusion(self, message: DataMessage) -> Dict[str, Any]:
        """Handle data fusion request."""
        try:
            data_type = message.data_type
            
            # Store incoming data
            await self._store_data(message)
            
            # Perform fusion if we have sufficient data
            fusion_result = await self._perform_fusion(data_type)
            
            # Record fusion result
            self.fusion_results.append({
                'timestamp': datetime.now(),
                'data_type': data_type,
                'fusion_result': fusion_result,
                'source_count': len(self.data_sources.get(data_type, []))
            })
            
            return fusion_result
            
        except Exception as e:
            return {
                'status': 'error',
                'data_type': getattr(message, 'data_type', 'unknown'),
                'error': str(e)
            }
    
    async def _store_data(self, message: DataMessage) -> None:
        """Store incoming data by type."""
        data_type = message.data_type
        
        if data_type not in self.data_sources:
            self.data_sources[data_type] = []
        
        data_entry = {
            'timestamp': datetime.now(),
            'source': message.source,
            'symbols': message.symbols,
            'payload': message.data_payload,
            'quality_score': message.quality_score
        }
        
        self.data_sources[data_type].append(data_entry)
        
        # Keep only recent data (last 100 entries per type)
        if len(self.data_sources[data_type]) > 100:
            self.data_sources[data_type] = self.data_sources[data_type][-100:]
    
    async def _perform_fusion(self, data_type: str) -> Dict[str, Any]:
        """Perform multimodal data fusion."""
        data_entries = self.data_sources.get(data_type, [])
        
        if not data_entries:
            return {
                'status': 'insufficient_data',
                'data_type': data_type,
                'message': 'No data available for fusion'
            }
        
        # Simulate fusion process
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Apply fusion algorithms
        fusion_results = {}
        
        for algorithm in self.fusion_algorithms:
            fusion_results[algorithm] = await self._apply_fusion_algorithm(
                algorithm, data_entries
            )
        
        # Generate consensus result
        consensus = await self._generate_consensus(fusion_results)
        
        return {
            'status': 'completed',
            'data_type': data_type,
            'fusion_timestamp': datetime.now(),
            'algorithm_results': fusion_results,
            'consensus': consensus,
            'confidence_score': consensus.get('confidence', 0.0),
            'data_sources_count': len(data_entries)
        }
    
    async def _apply_fusion_algorithm(
        self, 
        algorithm: str, 
        data_entries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Apply specific fusion algorithm."""
        import random
        
        if algorithm == 'weighted_average':
            # Simulate weighted average fusion
            weights = [entry.get('quality_score', 0.5) for entry in data_entries]
            total_weight = sum(weights) if weights else 1.0
            
            return {
                'method': 'weighted_average',
                'result_score': random.uniform(0.3, 0.9),
                'confidence': total_weight / len(data_entries) if data_entries else 0.0,
                'weight_distribution': weights
            }
        
        elif algorithm == 'attention':
            # Simulate attention-based fusion
            attention_scores = [random.uniform(0.1, 1.0) for _ in data_entries]
            
            return {
                'method': 'attention',
                'result_score': random.uniform(0.4, 0.95),
                'confidence': max(attention_scores) if attention_scores else 0.0,
                'attention_weights': attention_scores
            }
        
        elif algorithm == 'ensemble':
            # Simulate ensemble fusion
            ensemble_predictions = [random.uniform(0.2, 0.8) for _ in data_entries]
            
            return {
                'method': 'ensemble',
                'result_score': sum(ensemble_predictions) / len(ensemble_predictions) if ensemble_predictions else 0.0,
                'confidence': 1.0 - (max(ensemble_predictions) - min(ensemble_predictions)) if ensemble_predictions else 0.0,
                'individual_predictions': ensemble_predictions
            }
        
        else:
            return {
                'method': algorithm,
                'result_score': random.uniform(0.3, 0.7),
                'confidence': 0.5,
                'error': f'Unknown algorithm: {algorithm}'
            }
    
    async def _generate_consensus(self, fusion_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate consensus from multiple fusion algorithms."""
        if not fusion_results:
            return {'consensus_score': 0.0, 'confidence': 0.0}
        
        # Calculate consensus score
        scores = [result.get('result_score', 0.0) for result in fusion_results.values()]
        confidences = [result.get('confidence', 0.0) for result in fusion_results.values()]
        
        consensus_score = sum(scores) / len(scores) if scores else 0.0
        consensus_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Calculate agreement level
        score_variance = sum((s - consensus_score) ** 2 for s in scores) / len(scores) if scores else 0.0
        agreement_level = max(0.0, 1.0 - score_variance)
        
        return {
            'consensus_score': consensus_score,
            'confidence': consensus_confidence,
            'agreement_level': agreement_level,
            'algorithm_count': len(fusion_results),
            'score_variance': score_variance
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
            if isinstance(message, DataMessage):
                result = await self._handle_data_fusion(message)
                return QuantMessage(
                    message_type=MessageType.DATA_RESPONSE,
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
    
    def get_data_sources_summary(self) -> Dict[str, int]:
        """Get summary of available data sources."""
        return {
            data_type: len(entries) 
            for data_type, entries in self.data_sources.items()
        }
    
    def get_fusion_history(self) -> List[Dict[str, Any]]:
        """Get fusion history."""
        return self.fusion_results.copy()
    
    def get_latest_fusion_result(self, data_type: str = None) -> Optional[Dict[str, Any]]:
        """Get latest fusion result for a specific data type."""
        if not self.fusion_results:
            return None
        
        if data_type:
            # Find latest result for specific data type
            for result in reversed(self.fusion_results):
                if result.get('data_type') == data_type:
                    return result
            return None
        else:
            # Return latest overall result
            return self.fusion_results[-1]

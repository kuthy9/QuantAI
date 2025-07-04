"""
Feedback Loop Agent for the QuantAI multi-agent system.

This agent is responsible for learning from strategy performance,
collecting feedback, and improving the system over time.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

from autogen_core import Agent, MessageContext
from ...core.base import BaseQuantAgent, AgentRole, AgentCapability
from ...core.messages import QuantMessage, MessageType, FeedbackMessage, StrategyMessage


class FeedbackLoopAgent(BaseQuantAgent):
    """
    Agent responsible for learning from performance feedback.
    
    This agent collects and analyzes performance feedback to:
    - Identify successful strategy patterns
    - Learn from failures and mistakes
    - Suggest improvements to strategies
    - Update system parameters based on performance
    - Maintain a knowledge base of lessons learned
    """
    
    def __init__(self, agent_id: str = "feedback_loop"):
        super().__init__(
            role=AgentRole.FEEDBACK_LOOP,
            capabilities=[
                AgentCapability.LEARNING,
                AgentCapability.PERFORMANCE_ANALYSIS,
                AgentCapability.KNOWLEDGE_MANAGEMENT,
            ]
        )
        self.agent_id = agent_id
        self.feedback_history: List[Dict[str, Any]] = []
        self.lessons_learned: List[Dict[str, Any]] = []
        self.success_patterns: List[Dict[str, Any]] = []
        self.failure_patterns: List[Dict[str, Any]] = []
        self.improvement_suggestions: List[Dict[str, Any]] = []
        self.learning_metrics = {
            'total_feedback_processed': 0,
            'successful_strategies': 0,
            'failed_strategies': 0,
            'lessons_learned_count': 0,
            'improvement_suggestions_count': 0
        }
    
    async def on_messages(self, messages: List[QuantMessage], ctx: MessageContext) -> str:
        """Handle incoming messages for feedback processing."""
        results = []
        
        for message in messages:
            if isinstance(message, FeedbackMessage):
                result = await self._handle_feedback_processing(message)
                results.append(result)
            else:
                result = await self._handle_general_message(message)
                results.append(result)
        
        return f"FeedbackLoopAgent processed {len(results)} messages"
    
    async def _handle_feedback_processing(self, message: FeedbackMessage) -> Dict[str, Any]:
        """Handle feedback processing request."""
        try:
            strategy_id = message.strategy_id
            
            # Process the feedback
            processing_result = await self._process_feedback(message)
            
            # Extract lessons learned
            lessons = await self._extract_lessons(message)
            
            # Generate improvement suggestions
            suggestions = await self._generate_improvements(message)
            
            # Update knowledge base
            await self._update_knowledge_base(message, lessons, suggestions)
            
            # Record feedback processing
            self.feedback_history.append({
                'strategy_id': strategy_id,
                'processing_time': datetime.now(),
                'processing_result': processing_result,
                'lessons_count': len(lessons),
                'suggestions_count': len(suggestions)
            })
            
            # Update metrics
            self._update_learning_metrics(message)
            
            return {
                'status': 'processed',
                'strategy_id': strategy_id,
                'lessons_learned': lessons,
                'improvement_suggestions': suggestions,
                'processing_result': processing_result
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'strategy_id': getattr(message, 'strategy_id', 'unknown'),
                'error': str(e)
            }
    
    async def _process_feedback(self, feedback: FeedbackMessage) -> Dict[str, Any]:
        """Process performance feedback."""
        strategy_id = feedback.strategy_id
        actual_performance = feedback.performance_actual
        expected_performance = feedback.performance_expected
        
        # Calculate performance metrics
        performance_analysis = await self._analyze_performance(
            actual_performance, expected_performance
        )
        
        # Classify performance
        performance_classification = await self._classify_performance(performance_analysis)
        
        # Identify key factors
        key_factors = await self._identify_key_factors(feedback)
        
        return {
            'strategy_id': strategy_id,
            'performance_analysis': performance_analysis,
            'classification': performance_classification,
            'key_factors': key_factors,
            'processing_timestamp': datetime.now()
        }
    
    async def _analyze_performance(
        self, 
        actual: Dict[str, float], 
        expected: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze performance differences."""
        analysis = {
            'metrics_comparison': {},
            'overall_performance': 'unknown',
            'performance_score': 0.0,
            'variance_analysis': {}
        }
        
        # Compare each metric
        total_score = 0.0
        metric_count = 0
        
        for metric, expected_value in expected.items():
            if metric in actual:
                actual_value = actual[metric]
                difference = actual_value - expected_value
                relative_difference = difference / expected_value if expected_value != 0 else 0.0
                
                analysis['metrics_comparison'][metric] = {
                    'expected': expected_value,
                    'actual': actual_value,
                    'difference': difference,
                    'relative_difference': relative_difference,
                    'performance': 'better' if difference > 0 else 'worse' if difference < 0 else 'equal'
                }
                
                # Calculate metric score (1.0 = met expectations, >1.0 = exceeded, <1.0 = underperformed)
                metric_score = actual_value / expected_value if expected_value != 0 else 1.0
                total_score += metric_score
                metric_count += 1
        
        # Calculate overall performance score
        if metric_count > 0:
            analysis['performance_score'] = total_score / metric_count
            
            if analysis['performance_score'] >= 1.1:
                analysis['overall_performance'] = 'excellent'
            elif analysis['performance_score'] >= 1.0:
                analysis['overall_performance'] = 'good'
            elif analysis['performance_score'] >= 0.9:
                analysis['overall_performance'] = 'acceptable'
            else:
                analysis['overall_performance'] = 'poor'
        
        return analysis
    
    async def _classify_performance(self, analysis: Dict[str, Any]) -> str:
        """Classify performance into categories."""
        performance_score = analysis.get('performance_score', 0.0)
        
        if performance_score >= 1.2:
            return 'outstanding_success'
        elif performance_score >= 1.0:
            return 'success'
        elif performance_score >= 0.8:
            return 'moderate_success'
        elif performance_score >= 0.6:
            return 'underperformance'
        else:
            return 'failure'
    
    async def _identify_key_factors(self, feedback: FeedbackMessage) -> Dict[str, List[str]]:
        """Identify key success and failure factors."""
        return {
            'success_factors': feedback.success_factors or [],
            'failure_factors': feedback.failure_factors or [],
            'external_factors': [],  # Could be extracted from market conditions
            'strategy_factors': []   # Could be extracted from strategy parameters
        }
    
    async def _extract_lessons(self, feedback: FeedbackMessage) -> List[Dict[str, Any]]:
        """Extract lessons learned from feedback."""
        lessons = []
        
        # Process explicit lessons from feedback
        for lesson_text in feedback.lessons_learned or []:
            lesson = {
                'lesson_id': f"lesson_{len(self.lessons_learned) + len(lessons) + 1}",
                'strategy_id': feedback.strategy_id,
                'lesson_text': lesson_text,
                'lesson_type': 'explicit',
                'confidence': 0.8,
                'timestamp': datetime.now(),
                'source': 'feedback_message'
            }
            lessons.append(lesson)
        
        # Extract implicit lessons from performance analysis
        actual = feedback.performance_actual
        expected = feedback.performance_expected
        
        for metric, expected_value in expected.items():
            if metric in actual:
                actual_value = actual[metric]
                if actual_value < expected_value * 0.8:  # Significant underperformance
                    lesson = {
                        'lesson_id': f"lesson_{len(self.lessons_learned) + len(lessons) + 1}",
                        'strategy_id': feedback.strategy_id,
                        'lesson_text': f"Strategy underperformed in {metric}: expected {expected_value:.3f}, got {actual_value:.3f}",
                        'lesson_type': 'implicit_failure',
                        'confidence': 0.6,
                        'timestamp': datetime.now(),
                        'source': 'performance_analysis',
                        'metric': metric,
                        'performance_gap': expected_value - actual_value
                    }
                    lessons.append(lesson)
                elif actual_value > expected_value * 1.2:  # Significant overperformance
                    lesson = {
                        'lesson_id': f"lesson_{len(self.lessons_learned) + len(lessons) + 1}",
                        'strategy_id': feedback.strategy_id,
                        'lesson_text': f"Strategy excelled in {metric}: expected {expected_value:.3f}, achieved {actual_value:.3f}",
                        'lesson_type': 'implicit_success',
                        'confidence': 0.7,
                        'timestamp': datetime.now(),
                        'source': 'performance_analysis',
                        'metric': metric,
                        'performance_gain': actual_value - expected_value
                    }
                    lessons.append(lesson)
        
        return lessons
    
    async def _generate_improvements(self, feedback: FeedbackMessage) -> List[Dict[str, Any]]:
        """Generate improvement suggestions based on feedback."""
        suggestions = []
        
        # Process explicit suggestions from feedback
        for suggestion_text in feedback.improvement_suggestions or []:
            suggestion = {
                'suggestion_id': f"suggestion_{len(self.improvement_suggestions) + len(suggestions) + 1}",
                'strategy_id': feedback.strategy_id,
                'suggestion_text': suggestion_text,
                'suggestion_type': 'explicit',
                'priority': 'medium',
                'confidence': 0.8,
                'timestamp': datetime.now(),
                'source': 'feedback_message'
            }
            suggestions.append(suggestion)
        
        # Generate suggestions based on failure factors
        for failure_factor in feedback.failure_factors or []:
            suggestion = {
                'suggestion_id': f"suggestion_{len(self.improvement_suggestions) + len(suggestions) + 1}",
                'strategy_id': feedback.strategy_id,
                'suggestion_text': f"Address failure factor: {failure_factor}",
                'suggestion_type': 'failure_mitigation',
                'priority': 'high',
                'confidence': 0.7,
                'timestamp': datetime.now(),
                'source': 'failure_analysis',
                'related_factor': failure_factor
            }
            suggestions.append(suggestion)
        
        # Generate suggestions based on performance gaps
        actual = feedback.performance_actual
        expected = feedback.performance_expected
        
        for metric, expected_value in expected.items():
            if metric in actual:
                actual_value = actual[metric]
                if actual_value < expected_value * 0.9:  # Underperformance
                    suggestion = {
                        'suggestion_id': f"suggestion_{len(self.improvement_suggestions) + len(suggestions) + 1}",
                        'strategy_id': feedback.strategy_id,
                        'suggestion_text': f"Improve {metric} performance: current {actual_value:.3f}, target {expected_value:.3f}",
                        'suggestion_type': 'performance_improvement',
                        'priority': 'medium' if actual_value >= expected_value * 0.8 else 'high',
                        'confidence': 0.6,
                        'timestamp': datetime.now(),
                        'source': 'performance_gap_analysis',
                        'metric': metric,
                        'target_improvement': expected_value - actual_value
                    }
                    suggestions.append(suggestion)
        
        return suggestions
    
    async def _update_knowledge_base(
        self, 
        feedback: FeedbackMessage, 
        lessons: List[Dict[str, Any]], 
        suggestions: List[Dict[str, Any]]
    ) -> None:
        """Update the knowledge base with new lessons and suggestions."""
        # Add lessons to knowledge base
        self.lessons_learned.extend(lessons)
        
        # Add suggestions to knowledge base
        self.improvement_suggestions.extend(suggestions)
        
        # Update success/failure patterns
        performance_analysis = await self._analyze_performance(
            feedback.performance_actual, feedback.performance_expected
        )
        
        if performance_analysis['performance_score'] >= 1.0:
            # Success pattern
            pattern = {
                'pattern_id': f"success_{len(self.success_patterns) + 1}",
                'strategy_id': feedback.strategy_id,
                'success_factors': feedback.success_factors or [],
                'performance_metrics': feedback.performance_actual,
                'timestamp': datetime.now(),
                'confidence': min(1.0, performance_analysis['performance_score'] - 1.0 + 0.5)
            }
            self.success_patterns.append(pattern)
        else:
            # Failure pattern
            pattern = {
                'pattern_id': f"failure_{len(self.failure_patterns) + 1}",
                'strategy_id': feedback.strategy_id,
                'failure_factors': feedback.failure_factors or [],
                'performance_metrics': feedback.performance_actual,
                'timestamp': datetime.now(),
                'severity': 1.0 - performance_analysis['performance_score']
            }
            self.failure_patterns.append(pattern)
        
        # Keep knowledge base size manageable
        max_entries = 1000
        if len(self.lessons_learned) > max_entries:
            self.lessons_learned = self.lessons_learned[-max_entries:]
        if len(self.improvement_suggestions) > max_entries:
            self.improvement_suggestions = self.improvement_suggestions[-max_entries:]
        if len(self.success_patterns) > max_entries // 2:
            self.success_patterns = self.success_patterns[-max_entries // 2:]
        if len(self.failure_patterns) > max_entries // 2:
            self.failure_patterns = self.failure_patterns[-max_entries // 2:]
    
    def _update_learning_metrics(self, feedback: FeedbackMessage) -> None:
        """Update learning metrics."""
        self.learning_metrics['total_feedback_processed'] += 1
        
        # Determine if strategy was successful
        actual = feedback.performance_actual
        expected = feedback.performance_expected
        
        if actual and expected:
            # Calculate average performance ratio
            ratios = []
            for metric in expected:
                if metric in actual and expected[metric] != 0:
                    ratios.append(actual[metric] / expected[metric])
            
            if ratios:
                avg_ratio = sum(ratios) / len(ratios)
                if avg_ratio >= 1.0:
                    self.learning_metrics['successful_strategies'] += 1
                else:
                    self.learning_metrics['failed_strategies'] += 1
    
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
            if isinstance(message, FeedbackMessage):
                result = await self._handle_feedback_processing(message)
                return QuantMessage(
                    message_type=MessageType.FEEDBACK_RESPONSE,
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
    
    def get_lessons_learned(self, strategy_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get lessons learned, optionally filtered by strategy."""
        lessons = self.lessons_learned
        
        if strategy_id:
            lessons = [l for l in lessons if l.get('strategy_id') == strategy_id]
        
        return lessons[-limit:] if limit else lessons
    
    def get_improvement_suggestions(self, strategy_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get improvement suggestions, optionally filtered by strategy."""
        suggestions = self.improvement_suggestions
        
        if strategy_id:
            suggestions = [s for s in suggestions if s.get('strategy_id') == strategy_id]
        
        return suggestions[-limit:] if limit else suggestions
    
    def get_success_patterns(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get identified success patterns."""
        return self.success_patterns[-limit:] if limit else self.success_patterns
    
    def get_failure_patterns(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get identified failure patterns."""
        return self.failure_patterns[-limit:] if limit else self.failure_patterns
    
    def get_learning_metrics(self) -> Dict[str, Any]:
        """Get learning metrics and statistics."""
        metrics = self.learning_metrics.copy()
        metrics['lessons_learned_count'] = len(self.lessons_learned)
        metrics['improvement_suggestions_count'] = len(self.improvement_suggestions)
        metrics['success_patterns_count'] = len(self.success_patterns)
        metrics['failure_patterns_count'] = len(self.failure_patterns)
        
        # Calculate success rate
        total_strategies = metrics['successful_strategies'] + metrics['failed_strategies']
        metrics['success_rate'] = metrics['successful_strategies'] / total_strategies if total_strategies > 0 else 0.0
        
        return metrics

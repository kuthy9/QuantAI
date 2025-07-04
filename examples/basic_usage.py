"""
Basic usage example for the QuantAI AutoGen system.

This example demonstrates how to set up and run the multi-agent system
for automated quantitative trading strategy development.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from quantai.core.runtime import QuantRuntime, get_runtime
from quantai.core.config import QuantConfig
from quantai.core.messages import DataMessage, MessageType
from quantai.agents.data import DataIngestionAgent, MultimodalFusionAgent
from quantai.agents.analysis import MacroInsightAgent, StrategyGenerationAgent, StrategyCodingAgent
from quantai.agents.validation import StrategyValidatorAgent, RiskControlAgent
from quantai.core.base import AgentRole


async def setup_system() -> QuantRuntime:
    """Set up the QuantAI system with all agents."""
    
    print("🚀 Initializing QuantAI AutoGen System...")
    
    # Load configuration
    config = QuantConfig()
    
    # Create runtime
    runtime = QuantRuntime(config)
    
    # Register all agent types
    print("📝 Registering agents...")
    
    await runtime.register_agent_type(AgentRole.DATA_INGESTION, DataIngestionAgent)
    await runtime.register_agent_type(AgentRole.MULTIMODAL_FUSION, MultimodalFusionAgent)
    await runtime.register_agent_type(AgentRole.MACRO_INSIGHT, MacroInsightAgent)
    await runtime.register_agent_type(AgentRole.STRATEGY_GENERATION, StrategyGenerationAgent)
    await runtime.register_agent_type(AgentRole.STRATEGY_CODING, StrategyCodingAgent)
    await runtime.register_agent_type(AgentRole.STRATEGY_VALIDATION, StrategyValidatorAgent)
    await runtime.register_agent_type(AgentRole.RISK_CONTROL, RiskControlAgent)
    
    print("✅ All agents registered successfully")
    
    # Start the runtime
    await runtime.start()
    print("🎯 Runtime started successfully")
    
    return runtime


async def demonstrate_data_pipeline(runtime: QuantRuntime):
    """Demonstrate the data collection and processing pipeline."""
    
    print("\n📊 Demonstrating Data Pipeline...")
    
    # Request market data
    market_request = DataMessage(
        message_type=MessageType.DATA_REQUEST,
        sender_id="demo_user",
        data_type="market",
        symbols=["SPY", "QQQ", "IWM", "VIX"],
    )
    
    print("📈 Requesting market data for SPY, QQQ, IWM, VIX...")
    await runtime.send_message(market_request)
    
    # Request news data
    news_request = DataMessage(
        message_type=MessageType.DATA_REQUEST,
        sender_id="demo_user",
        data_type="news",
        symbols=["SPY", "QQQ"],
    )
    
    print("📰 Requesting news data...")
    await runtime.send_message(news_request)
    
    # Wait for data processing
    print("⏳ Waiting for data processing...")
    await asyncio.sleep(10)
    
    print("✅ Data pipeline demonstration completed")


async def demonstrate_strategy_workflow(runtime: QuantRuntime):
    """Demonstrate the complete strategy development workflow."""
    
    print("\n🧠 Demonstrating Strategy Development Workflow...")
    
    # The workflow will be triggered by the data pipeline
    # In a real system, this would happen automatically through message routing
    
    print("🔄 Strategy workflow initiated by data signals...")
    print("   1. Data Ingestion → Multimodal Fusion")
    print("   2. Multimodal Fusion → Macro Analysis")
    print("   3. Macro Analysis → Strategy Generation")
    print("   4. Strategy Generation → Strategy Coding")
    print("   5. Strategy Coding → Strategy Validation")
    print("   6. Strategy Validation → Risk Assessment")
    
    # Wait for the workflow to process
    await asyncio.sleep(30)
    
    print("✅ Strategy development workflow demonstration completed")


async def monitor_system_status(runtime: QuantRuntime):
    """Monitor and display system status."""
    
    print("\n📊 System Status Monitor...")
    
    # Get system status
    status = await runtime.get_system_status()
    
    print(f"Runtime Status: {status['runtime_status']}")
    print(f"Total Agents: {status['total_agents']}")
    print(f"Uptime: {status['uptime_seconds']:.1f} seconds")
    
    print("\nAgents by Role:")
    for role, count in status['agents_by_role'].items():
        if count > 0:
            print(f"  {role}: {count}")
    
    # Health check
    health = await runtime.health_check()
    print(f"\nSystem Health: {health['status']}")
    if health['unhealthy_agents']:
        print(f"Unhealthy Agents: {health['unhealthy_agents']}")


async def demonstrate_risk_monitoring():
    """Demonstrate risk monitoring capabilities."""
    
    print("\n🛡️ Demonstrating Risk Monitoring...")
    
    # This would typically be triggered by actual trading activity
    # For demo purposes, we'll show the concept
    
    print("Risk monitoring includes:")
    print("  • Position size limits")
    print("  • Portfolio leverage monitoring")
    print("  • Drawdown tracking")
    print("  • Correlation analysis")
    print("  • VaR calculation")
    print("  • Emergency stop capabilities")
    
    print("✅ Risk monitoring demonstration completed")


async def demonstrate_memory_system():
    """Demonstrate the memory and learning system."""
    
    print("\n🧠 Demonstrating Memory System...")
    
    print("Memory system capabilities:")
    print("  • Strategy performance tracking")
    print("  • Market regime memory")
    print("  • Failure analysis and learning")
    print("  • Cross-modal signal correlation")
    print("  • Long-term pattern recognition")
    
    print("✅ Memory system demonstration completed")


async def cleanup_system(runtime: QuantRuntime):
    """Clean up and shutdown the system."""
    
    print("\n🔄 Shutting down system...")
    
    # Get final status
    final_status = await runtime.get_system_status()
    print(f"Final uptime: {final_status['uptime_seconds']:.1f} seconds")
    
    # Shutdown runtime
    await runtime.stop()
    print("✅ System shutdown completed")


async def main():
    """Main demonstration function."""
    
    print("=" * 60)
    print("🤖 QuantAI AutoGen - Multi-Agent Financial System Demo")
    print("=" * 60)
    
    runtime = None
    
    try:
        # Setup system
        runtime = await setup_system()
        
        # Demonstrate various capabilities
        await demonstrate_data_pipeline(runtime)
        await demonstrate_strategy_workflow(runtime)
        await monitor_system_status(runtime)
        await demonstrate_risk_monitoring()
        await demonstrate_memory_system()
        
        print("\n" + "=" * 60)
        print("🎉 Demo completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        
    finally:
        if runtime:
            await cleanup_system(runtime)


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())

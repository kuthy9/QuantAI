#!/usr/bin/env python3
"""
Production Simulation Script for QuantAI AutoGen System.

This script runs a comprehensive end-to-end production simulation with:
- IBKR simulation account integration (DUA559603)
- Telegram notifications
- Real-time performance monitoring
- Enhanced performance targets (>25% annual return, >1.5 Sharpe ratio, <10% max drawdown)

Usage:
    python run_production_simulation.py [--duration HOURS] [--capital AMOUNT]
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
from src.quantai.simulation.production_runner import ProductionSimulationRunner, SimulationConfig


def setup_environment():
    """Setup environment and validate configuration."""
    # Check required environment variables
    required_vars = [
        "TELEGRAM_BOT_TOKEN",
        "TELEGRAM_CHAT_ID", 
        "IB_ACCOUNT",
        "OPENAI_API_KEY"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.info("Please check your .env file and ensure all required variables are set")
        return False
    
    # Validate IBKR account
    ib_account = os.getenv("IB_ACCOUNT")
    if ib_account != "DUA559603":
        logger.warning(f"IB_ACCOUNT is {ib_account}, expected DUA559603")
    
    # Validate Telegram credentials
    telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
    telegram_chat = os.getenv("TELEGRAM_CHAT_ID")
    
    if not telegram_token.startswith("8139833297:"):
        logger.warning("Telegram bot token doesn't match expected format")
    
    if telegram_chat != "7871286497":
        logger.warning(f"Telegram chat ID is {telegram_chat}, expected 7871286497")
    
    logger.info("Environment validation completed")
    return True


async def run_simulation(duration_hours: int = 24, initial_capital: float = 100000.0):
    """Run the production simulation."""
    
    # Create simulation configuration with enhanced targets
    config = SimulationConfig(
        duration_hours=duration_hours,
        target_annual_return=0.25,  # 25% annual return target
        target_sharpe_ratio=1.5,    # Enhanced Sharpe ratio target
        max_drawdown=0.10,          # 10% max drawdown limit
        initial_capital=initial_capital,
        max_position_size=0.1,      # 10% max position size
        max_daily_trades=50,
        risk_check_interval=300,    # 5 minutes
        performance_update_interval=3600,  # 1 hour
        log_level="INFO",
        save_results=True
    )
    
    logger.info("=" * 80)
    logger.info("QUANTAI AUTOGEN PRODUCTION SIMULATION")
    logger.info("=" * 80)
    logger.info(f"Duration: {duration_hours} hours")
    logger.info(f"Initial Capital: ${initial_capital:,.2f}")
    logger.info(f"Target Annual Return: {config.target_annual_return:.1%}")
    logger.info(f"Target Sharpe Ratio: {config.target_sharpe_ratio:.1f}")
    logger.info(f"Max Drawdown Limit: {config.max_drawdown:.1%}")
    logger.info(f"IBKR Account: {os.getenv('IB_ACCOUNT')}")
    logger.info(f"Telegram Chat: {os.getenv('TELEGRAM_CHAT_ID')}")
    logger.info("=" * 80)
    
    # Create and run simulation
    runner = ProductionSimulationRunner(config)
    
    try:
        # Run the simulation
        results = await runner.run_simulation()
        
        # Display results
        print_simulation_results(results)
        
        return results
        
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        await runner.stop_simulation()
        return None
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise


def print_simulation_results(results: dict):
    """Print formatted simulation results."""
    metrics = results["performance_metrics"]
    success_criteria = results["success_criteria"]
    
    print("\n" + "=" * 80)
    print("SIMULATION RESULTS")
    print("=" * 80)
    
    # Performance Summary
    print(f"Duration: {metrics['duration_hours']:.1f} hours")
    print(f"Initial Capital: ${metrics['initial_capital']:,.2f}")
    print(f"Final Capital: ${metrics['current_capital']:,.2f}")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Annualized Return: {metrics['annual_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Volatility: {metrics['volatility']:.2%}")
    
    print("\nTrading Metrics:")
    print(f"Total Trades: {metrics['total_trades']}")
    print(f"Winning Trades: {metrics['winning_trades']}")
    print(f"Losing Trades: {metrics['losing_trades']}")
    print(f"Win Rate: {metrics['win_rate']:.1%}")
    
    print("\nSystem Metrics:")
    print(f"Agent Errors: {metrics['agent_errors']}")
    print(f"Connection Issues: {metrics['connection_issues']}")
    print(f"Uptime: {metrics['uptime_percentage']:.1f}%")
    
    print("\nSuccess Criteria:")
    print(f"Annual Return Target: {success_criteria['annual_return_target']:.1%} "
          f"({'✓' if success_criteria['annual_return_met'] else '✗'})")
    print(f"Sharpe Ratio Target: {success_criteria['sharpe_target']:.1f} "
          f"({'✓' if success_criteria['sharpe_met'] else '✗'})")
    print(f"Drawdown Limit: {success_criteria['drawdown_limit']:.1%} "
          f"({'✓' if success_criteria['drawdown_met'] else '✗'})")
    
    overall_success = success_criteria['overall_success']
    print(f"\nOVERALL SUCCESS: {'✓ PASSED' if overall_success else '✗ FAILED'}")
    
    if overall_success:
        print("🎉 Congratulations! The QuantAI system has successfully met all performance targets.")
    else:
        print("⚠️  The system did not meet all performance targets. Review the metrics above.")
    
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run QuantAI AutoGen Production Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_production_simulation.py                    # Run 24-hour simulation
  python run_production_simulation.py --duration 8      # Run 8-hour simulation  
  python run_production_simulation.py --capital 50000   # Run with $50K capital
        """
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=24,
        help="Simulation duration in hours (default: 24)"
    )
    
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital amount (default: 100000)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger.remove()
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # Validate environment
    if not setup_environment():
        sys.exit(1)
    
    # Run simulation
    try:
        results = asyncio.run(run_simulation(args.duration, args.capital))
        
        if results and results["success_criteria"]["overall_success"]:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failed to meet targets
            
    except Exception as e:
        logger.error(f"Simulation failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

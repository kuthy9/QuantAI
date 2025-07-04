# QuantAI AutoGen - Multi-Agent Financial Quantitative System

A comprehensive multi-agent AI system built on Microsoft AutoGen for automated financial quantitative algorithm generation, backtesting, and execution.

## 🏗️ Architecture Overview

The system consists of 16 specialized agents organized into functional layers:

### 📡 Data Layer
- **D1 - Data Ingestion Agent**: Crawls market data, news, earnings, and sentiment data
- **D4 - Multimodal Fusion Agent**: Fuses structured data, text, and image signals

### 🧠 Analysis Layer  
- **A0 - Macro Insight Agent**: Analyzes macro trends, cycles, and market regimes
- **A1 - Strategy Generation Agent**: Generates multiple draft strategies
- **A2 - Strategy Coding Agent**: Converts strategies to executable code (Lean platform)

### 🧪 Validation Layer
- **D5 - Strategy Validator Agent**: Audits code for logic issues, overfitting, data leakage
- **D2 - Risk Control Agent**: Monitors exposure, leverage, volatility, drawdown

### 🚀 Execution Layer
- **A3 - Strategy Deployment Agent**: Deploys to backtest/paper/live modes
- **D3 - Execution Agent**: Optimizes and executes trade signals in real-time
- **A4 - Backtest Monitor Agent**: Backtests and returns performance metrics

### 🔄 Learning Layer
- **A6 - Profitability Agent**: Decides whether to go live based on performance
- **A5 - Feedback Loop Agent**: Learns from failed strategies
- **D6 - Long-term Memory Agent**: Stores strategy results and insights

### 🔐 Control Layer
- **M1 - API Key + Broker Account Manager**: Manages credentials and accounts
- **M3 - Kill Switch Agent**: Emergency stop and forced liquidation
- **V0 - Visual Dashboard**: Web-based monitoring and control interface

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/quantai/quantai-autogen.git
cd quantai-autogen

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Configure your API keys in `.env`:
```bash
# AI Model APIs
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key

# Trading APIs (for live trading)
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # Paper trading

# Data APIs
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
NEWSAPI_KEY=your_newsapi_key
```

### Basic Usage

```python
import asyncio
from quantai import QuantRuntime, get_config
from quantai.agents import *

async def main():
    # Initialize the runtime
    config = get_config()
    runtime = QuantRuntime(config)
    
    # Register agents
    await runtime.register_agent_type(AgentRole.DATA_INGESTION, DataIngestionAgent)
    await runtime.register_agent_type(AgentRole.MULTIMODAL_FUSION, MultimodalFusionAgent)
    await runtime.register_agent_type(AgentRole.MACRO_INSIGHT, MacroInsightAgent)
    await runtime.register_agent_type(AgentRole.STRATEGY_GENERATION, StrategyGenerationAgent)
    await runtime.register_agent_type(AgentRole.STRATEGY_CODING, StrategyCodingAgent)
    await runtime.register_agent_type(AgentRole.STRATEGY_VALIDATION, StrategyValidatorAgent)
    await runtime.register_agent_type(AgentRole.RISK_CONTROL, RiskControlAgent)
    
    # Start the system
    await runtime.start()
    
    # Request market data analysis
    from quantai.core.messages import DataMessage, MessageType
    
    data_request = DataMessage(
        message_type=MessageType.DATA_REQUEST,
        sender_id="user",
        data_type="market",
        symbols=["SPY", "QQQ", "IWM"],
    )
    
    await runtime.send_message(data_request)
    
    # Let the system run for a while
    await asyncio.sleep(60)
    
    # Get system status
    status = await runtime.get_system_status()
    print(f"System Status: {status}")
    
    # Shutdown
    await runtime.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

## 🔧 Key Features

### 🤖 Multi-Model AI Integration
- **OpenAI GPT-4**: Strategy generation and analysis
- **Anthropic Claude**: Code generation and validation  
- **Google Gemini**: Multimodal fusion and execution
- **Automatic fallback**: Seamless switching between models

### 📊 Comprehensive Data Sources
- **Market Data**: Real-time prices, volumes, technical indicators
- **News Analysis**: Sentiment analysis from financial news
- **Earnings Data**: Fundamental analysis and earnings surprises
- **Social Sentiment**: Social media and forum sentiment tracking

### 🧠 Advanced Strategy Generation
- **Multiple Strategy Types**: Momentum, mean reversion, arbitrage, factor-based
- **Regime-Aware**: Adapts to different market conditions
- **Risk-Adjusted**: Optimizes for Sharpe ratio and drawdown
- **Multi-Timeframe**: Intraday, daily, weekly, monthly strategies

### 🛡️ Robust Risk Management
- **Real-time Monitoring**: Continuous risk assessment
- **Position Limits**: Automatic position sizing and limits
- **Drawdown Controls**: Circuit breakers and emergency stops
- **Correlation Analysis**: Portfolio diversification monitoring

### 🔍 Comprehensive Validation
- **Code Quality**: Syntax, logic, and best practices validation
- **Overfitting Detection**: Statistical tests for robustness
- **Data Leakage**: Look-ahead bias and data integrity checks
- **Performance Validation**: Realistic expectation assessment

## 📈 Example Workflow

1. **Data Collection**: Agents continuously collect market data, news, and sentiment
2. **Signal Fusion**: Multimodal fusion creates unified market signals
3. **Macro Analysis**: Macro agent identifies current market regime
4. **Strategy Generation**: Multiple strategies generated based on market conditions
5. **Code Implementation**: Strategies converted to executable trading algorithms
6. **Validation**: Comprehensive validation for quality and risk
7. **Backtesting**: Historical performance testing and optimization
8. **Risk Assessment**: Portfolio risk analysis and approval
9. **Deployment**: Live trading with continuous monitoring
10. **Learning**: Performance feedback for continuous improvement

## 🏛️ Supported Trading Platforms

- **QuantConnect Lean**: Primary platform for algorithm development
- **Zipline**: Python backtesting framework
- **Backtrader**: Alternative Python trading framework
- **Alpaca**: Commission-free trading API
- **Interactive Brokers**: Professional trading platform

## 📊 Performance Metrics

The system tracks comprehensive performance metrics:

- **Returns**: Absolute and risk-adjusted returns
- **Risk Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown
- **Trading Metrics**: Win rate, profit factor, average trade
- **Risk Controls**: VaR, leverage, concentration, correlation
- **Operational**: Execution quality, slippage, transaction costs

## 🔒 Security & Compliance

- **API Key Management**: Secure credential storage and rotation
- **Risk Limits**: Configurable position and portfolio limits
- **Audit Trail**: Comprehensive logging of all decisions and trades
- **Emergency Controls**: Kill switch and emergency liquidation
- **Compliance**: Regulatory compliance monitoring

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_agents/
pytest tests/test_core/
pytest tests/test_integration/
```

## 📚 Documentation

- [Architecture Guide](docs/architecture.md)
- [Agent Development](docs/agents.md)
- [Configuration Reference](docs/configuration.md)
- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Trading financial instruments involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Always consult with a qualified financial advisor before making investment decisions.

## 🙏 Acknowledgments

- [Microsoft AutoGen](https://github.com/microsoft/autogen) - Multi-agent framework
- [QuantConnect](https://www.quantconnect.com/) - Algorithmic trading platform
- [OpenAI](https://openai.com/) - GPT models
- [Anthropic](https://www.anthropic.com/) - Claude models
- [Google](https://ai.google/) - Gemini models

---

**Built with ❤️ by the QuantAI Team**

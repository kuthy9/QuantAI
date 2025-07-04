# QuantAI AutoGen Agent Guide

## Commands
- **Test**: `pytest` (all tests), `pytest tests/test_agents/` (specific module), `pytest -k test_name` (single test)
- **Build**: `pip install -e .` (development install), `python -m quantai.cli` (CLI tool)
- **Lint**: `black src/ tests/` (formatting), `isort src/ tests/` (imports), `flake8 src/ tests/` (linting)
- **Type Check**: `mypy src/`
- **Coverage**: `pytest --cov=quantai --cov-report=html`
- **Setup**: `./setup.sh` (initial setup), `source activate_env.sh` (activate environment)

## Architecture
- **Multi-agent financial system** built on Microsoft AutoGen with 16 specialized agents
- **Core structure**: `src/quantai/` (main package), `src/quantai/agents/` (agent implementations), `src/quantai/core/` (base classes)
- **Layers**: Data (D1, D4), Analysis (A0-A2), Validation (D2, D5), Execution (A3, D3, A4), Learning (A5, A6, D6), Control (M1, M3, V0)
- **Database**: SQLAlchemy models, Redis for caching
- **API**: FastAPI web framework with WebSocket support
- **Key files**: `runtime.py` (system orchestration), `base.py` (agent base classes), `messages.py` (communication)

## Code Style
- **Python 3.10+** with type hints (`disallow_untyped_defs = true`)
- **Line length**: 100 characters (Black formatter)
- **Imports**: Use `isort` with Black profile, absolute imports preferred
- **Naming**: snake_case for functions/variables, PascalCase for classes, ALL_CAPS for constants
- **Error handling**: Use structured exceptions, async/await for I/O operations
- **Logging**: Use `loguru` logger, not print statements
- **Models**: Pydantic BaseModel for data validation, Enum for constants
- **Docstrings**: Google style with type information

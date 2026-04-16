# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Research Overview

**Dissertation Title**: Analysing the Potential Impacts of Tariffs, Trade and Regional Conflicts for Smart Investment Strategies in Different Time Zones Through Directional Changes and Large Language Models

### Research Background
- Recent surge in sudden policy changes: tariffs, trade wars, regional conflicts
- These policies significantly impact global supply chains, trading activities, and financial markets (stock prices, volumes)
- Different time zones exhibit varying immediate effects and impact intensities

### Key Concepts
- **Directional Changes (DC)**: Data-driven approach to detect significant/sudden movements in stock data based on threshold settings
- **Large Language Models (LLMs)**: ChatGPT, Gemini, DeepSeek, Grok for financial sentiment analysis
- **Multi-timezone Analysis**: Cross-market impact analysis across different regions

### Research Scope
| Category | Examples |
|----------|----------|
| A) Tariffs | Import taxes, duties on specific goods |
| B) Trade Wars | Sanctions, embargoes, trade restrictions |
| C) Regional Conflicts | Military conflicts, geopolitical events |
| Combined Effects | A+B (tariffs + trade war), etc. |

### Impact Characteristics
- **Immediate effects**: Sudden tariffs on specific imports affect markets instantly
- **Delayed effects**: Some policies take time to propagate through supply chains

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Run training (configure algorithm in config.py first)
python entrance.py
```

**Note**: TA-Lib is required and must be installed manually. See README.md for installation instructions.

## Configuration

Edit `config.py` to select algorithms and settings:
- `benchmark_algo`: Algorithm choice - 'MASA-dc', 'MASA-mlp', 'MASA-lstm', 'MASA-SentimentMulti', 'TD3-Profit', 'TD3-PR', 'TD3-SR', 'DC', 'CRP', 'EG', 'OLMAR', etc.
- `market_name`: 'SP500', 'DJIA'
- `topK`: Number of assets in portfolio (10, 20, 30)
- `num_epochs`: Training episodes
- `enable_market_observer`: Enable market observer module
- `enable_controller`: Enable CBF controller

## Architecture

### Three Execution Modes

1. **RLonly**: Single-agent RL (TD3) - standalone reinforcement learning
2. **RLcontroller**: MASA framework - RL agent + market observer + optional controller
3. **Benchmark**: Traditional portfolio optimization algorithms (DC, OLMAR, PAMR, etc.)

### Core Components

| Directory | Purpose |
|-----------|---------|
| `RL_controller/` | RL agent (TD3), market observer, CBF controller |
| `benchmark/` | Baseline algorithms (DC, EG, OLMAR, etc.) |
| `fin_sentiment/` | Financial sentiment generation from news |
| `utils/` | Feature generation, trading environment, model pool |
| `data/` | Stock price data (SP500, DJIA) |

### Data Flow

```
Price Data → Feature Generation → Trading Environment
                              ↓
                    Market Observer (optional)
                              ↓
                    RL Agent (TD3) / Benchmark Algo
                              ↓
                    Portfolio Weights → Performance Metrics
```

### Key Classes

- `StockPortfolioEnv` (utils/tradeEnv.py): Gym-style trading environment
- `MarketObserver` / `MarketObserver_Algorithmic` (RL_controller/market_obs.py): Generates market state embeddings and risk parameters
- `TD3PolicyAdj` (RL_controller/TD3_controller.py): Custom TD3 policy for portfolio optimization
- `FeatureProcesser` (utils/featGen.py): Generates technical indicators and DC features

## Output

Results are saved to `./res/{mode}/{model}/{market}-{topK}/{timestamp}/` including:
- Training/validation/test performance CSVs
- Model checkpoints
- Performance profile plots
# Intraday Algorithmic Trading using Reinforcement Learning
### Inter-IIT Tech Meet 14.0 — High-Frequency Trading Challenge

> **Note:** Strategy returns depend on the random train/test day split. High-return days like Day 87 (EBX) and Day 104 (EBY) can significantly boost results when included in the test set. To get reproducible or varied results, adjust the random seed via `PARAMS['SEED']`.

## Overview
This project implements an intraday **Reinforcement Learning (RL)** trading strategy for the Inter-IIT Tech Meet 14.0. The agent uses **Proximal Policy Optimization (PPO)** to learn trading decisions from high-frequency market data, driven by a feature-rich state space that includes technical indicators, Heikin-Ashi candlestick signals, and adaptive volatility filters.

The strategy was evaluated on two tickers — EBX and EBY — and delivered strong risk-adjusted returns across out-of-sample test periods.

### Backtested Performance
The following metrics are from out-of-sample backtesting as reported in the team's performance report:

| Metric | EBX (255 Days) | EBY (140 Days) |
| :--- | :--- | :--- |
| **Annualized Return** | **81.61%** | **77.97%** |
| **Calmar Ratio** | **70.96** | **63.91** |
| **Sharpe Ratio** | 7.30 | 4.91 |
| **Max Drawdown** | 1.15% | 1.22% |
| **Win Rate** | ~69% | ~69% |
| **Avg Trades/Day** | 13.77 | 18.85 |

---

## How It Works

### 1. Data Pipeline
* **Resampling:** Raw tick/second-level data is converted into **2-minute OHLC candles** to reduce microstructure noise while preserving intraday price action.
* **Input Features:**
    * **Trend:** Heikin-Ashi candle transformations, Johnny Ribbon for regime detection.
    * **Momentum:** RSI, CCI, CMO, Aroon oscillators.
    * **Volatility:** ATR, rolling Standard Deviation, Choppiness Index.
    * **Time Encoding:** Cyclical sine/cosine encoding to capture time-of-day effects.
    * **Adaptive Filter:** KAMA (Kaufman Adaptive Moving Average) for noise reduction.

### 2. PPO Agent Setup
* **Algorithm:** Proximal Policy Optimization via `stable-baselines3`.
* **Network:** `MlpPolicy` with two hidden layers of size [256, 256].
* **Reward Design:** Custom shaped reward that includes:
    * Scaling based on realized PnL.
    * Stop-loss penalty: `-100`; forced end-of-day exit penalty: `-10`.
    * Small bonus of `+0.1` for holding flat in choppy/uncertain conditions.
* **Training Setup:** Multiple parallel environments using `SubprocVecEnv`, normalized with `VecNormalize` for training stability.

---

## Getting Started
### Install Required Packages
```bash
pip install numpy pandas gymnasium stable-baselines3 torch tqdm matplotlib
```

### Data Setup
Organize your raw tick CSV files under a ticker-named folder (e.g., `EBX/`) and set that path via `PARAMS['SOURCE_FOLDER']` in the script:
```
EBX/
├── day1.csv
├── day2.csv
└── day3.csv
```

## Usage

### Step 1 — Train the Model: `python <Ticker>.py train`

**Steps performed:**

1. **Data Resampling** (2-3 mins)
   - Reads tick data from `EBX/` folder
   - Converts to 2-minute OHLC candles
   - Saves to `EBX_2min/` (skips if already exists)
   - Creates `train_days_EBX.txt` and `test_days_EBX.txt` by randomly selecting days

2. **Indicator Calculation** (1 min)
   - Precomputes 60+ technical indicators for ALL training days
   - Applies 30-minute warmup window (discards first 30 mins of each day)

3. **Model Training** (5-10 mins depending on CPU/GPU)
   - Launches parallel environments
   - Trains PPO model
   - Prints training progress with tqdm bar
   - Monitors: entropy loss, explained variance, policy loss
   - GPU auto-detects and uses if available

4. **Model Saving** 
   - Saves trained model: `Models_EBX/ppo_trading_model_EBX.zip`
   - Saves normalization stats: `Models_EBX/ppo_trading_model_EBX_vecnormalize.pkl`
   - Generates training plots: `training_plots/EBX_training_metrics.png`
   - Generates feature info: `feature_info_EBX.txt`

---

### Step 2 — Run Backtests: `python <Ticker>.py test`

**Steps performed:**

1. **Model Loading** 
   - Loads trained model from `Models_EBX/ppo_trading_model_EBX.zip`
   - Loads normalization stats from `Models_EBX/ppo_trading_model_EBX_vecnormalize.pkl`
   - Verifies both files exist

2. **Per-Day Backtesting** 
   - For each test day:
     - Loads 2-min candle data
     - Calculates indicators (with 30-min warmup)
     - Records every trade entry/exit with price and timestamp
     - Calculates daily P&L in basis points (bps)
     - Generates signals (BUY, SELL, EXIT)

3. **Output Generation** 
   - Saves all signals to CSV: `signals_EBX/day123.csv`
   - Generates price charts: `test_trade_plots/EBX_day_1_day123.png`
   - Calculates equity curve and drawdown
   - Saves equity plot: `test_results/EBX_equity_drawdown.png`
   - Writes report: `test_results/test_results_EBX.txt`
   - **Prints to console:** Trade log with timestamps, prices, positions

**Expected Bugs & Solutions:**

| Bug | Cause | Solution |
|-----|-------|----------|
| "VecNormalize file not found" | Didn't run train command first | Run `python <Ticker>.py train` first |
| All trades losing | Model overtrained on train set (overfitting) | Train on more diverse data or reduce training episodes |
| 0 trades executed | Model learned to always hold | Increase `TRADE_ENTRY_PENALTY` (currently -5) or check reward scaling |

---

### Step 3 — Test a Single Day: `python <Ticker>.py test 123`

**Steps performed:**

1. **Specific Day Filtering** 
   - Searches for `day123` in test file list
   - Only tests that single day (not all test days)
   - Useful for debugging specific days

---

### Step 4 — Official Simulator Backtest: `python <Ticker>.py backtest_ebullient`

**Steps performed:**

1. **Backtest Execution** 
   - Initializes BacktesterIIT with config
   - Runs Ebullient's market simulator
   - For each signal:
     - EXIT signal → Closes position
     - BUY signal → Opens long (100 shares)
     - SELL signal → Opens short (100 shares)
   - Prints backtest results

**Expected Bugs & Solutions:**

| Bug | Cause | Solution |
|-----|-------|----------|
| "day(\d+)" regex error | Signal file naming doesn't match pattern | Check files are named like `day1.csv`, `day2.csv` (not `day_1.csv`) |
| Config file error | JSON formatting issue | Manually inspect `config.json` created in root |

---

## Troubleshooting

### Issue 1: "PARAMS mismatch between training and testing"
**Problem:** You changed stop loss/trailing stop but didn't retrain
**Solution:** 
- Training uses `STOP_LOSS_TR`, `TRAIL_PCT_TR`
- Testing uses `STOP_LOSS_TE`, `TRAIL_PCT_TE` (can be different!)
- Model learns exits based on TRAINING params
- Testing params determine what exits are ENFORCED during test
- If you change testing params, results will differ (but model hasn't relearned)

### Issue 2: "Too many/too few trades"
**Problem:** Model behavior doesn't match expectations
**Solutions:**
- Too many: Increase `TRADE_ENTRY_PENALTY` (currently -5) to -10 or -15
- Too few: Decrease `TRADE_ENTRY_PENALTY` to 0 or -2
- Too many stops: Decrease `STOP_LOSS` from -0.0004 to -0.0002
- Retrain after changing parameters

---

## Expected Output Files

```
After train:
✓ Models_EBX/ppo_trading_model_EBX.zip (2-5MB)
✓ Models_EBX/ppo_trading_model_EBX_vecnormalize.pkl (100KB)
✓ feature_info_EBX.txt (50KB)
✓ train_days_EBX.txt (list of days)
✓ test_days_EBX.txt (list of days)
✓ training_plots/EBX_training_metrics.png (chart)
✓ EBX_2min/ (folder with 2-min candles - auto-created)

After test:
✓ test_results/test_results_EBX.txt (report)
✓ test_results/EBX_equity_drawdown.png (equity chart)
✓ test_trade_plots/ (folder with per-day charts)
✓ signals_EBX/ (folder with signal CSVs)
```

---

## Plots & Visualizations

The following plots are auto-generated and saved to `test_trade_plots/` and `training_plots/`:
* **Equity Curves:** Tracks cumulative portfolio value over the test period.
* **Drawdown Charts:** Shows peak-to-trough loss at each point in time.
* **Trade Charts:** Per-day candlestick charts with BUY/SELL/EXIT markers overlaid.
* **Training Diagnostics:** Entropy loss and explained variance to assess model convergence.

---

## Requirements

* Python 3.8+
* `numpy`
* `pandas`
* `gymnasium`
* `stable-baselines3`
* `torch`
* `matplotlib`
* `tqdm`





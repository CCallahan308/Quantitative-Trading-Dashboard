# 🔬 Experiments Explained

## What Are Experiments?

**Experiments** are systematic tests where you run the backtesting system with **different combinations of parameters** to find which settings work best.

Think of it like: "What if I use 100 trees instead of 500? What if I use 70% training data instead of 65%?"

## Two Types of Experiments

### 1. **Grid Search** (Test All Combinations)
You provide lists of values, and it tests **every possible combination**.

**Example:**
```
n_estimators: [100, 200, 500]           (3 options)
train_pct: [65, 70]                     (2 options)
val_pct: [15, 20]                       (2 options)
early_stopping: [0, 50]                 (2 options)

Total combinations: 3 × 2 × 2 × 2 = 24 experiments
```

Each experiment:
- Trains a separate XGBoost model
- Tests it on the same data
- Records results (ROC AUC, accuracy, final portfolio value)
- Saves to CSV

### 2. **Random Search** (Sample Randomly)
Instead of testing all combinations, randomly sample N combinations.

**Example:**
```
Trials: 10

This randomly picks 10 combinations from the parameter space
and tests those 10.
```

Use when you have too many parameters (would take too long otherwise).

## Where Experiments Go

### Location
```
📁 experiments/
   └── exp_YYYYMMDD_HHMMSS.csv
       └── exp_20251116_201413.csv      ← Each run gets a timestamped file
```

### Example File: `exp_20251116_201413.csv`

| n_estimators | train_pct | val_pct | early_stopping | roc_auc | accuracy | final_value |
|---|---|---|---|---|---|---|
| 100 | 65 | 15 | 0 | 0.473 | 0.456 | 1.771 |
| 100 | 65 | 15 | 50 | 0.473 | 0.456 | 1.771 |
| 100 | 65 | 20 | 0 | 0.457 | 0.463 | 1.771 |
| ... | ... | ... | ... | ... | ... | ... |
| 500 | 70 | 20 | 50 | 0.510 | 0.391 | 1.714 |

**Each row = One complete backtest with those parameters**

## How to Run Experiments (In the Dashboard)

### Step 1: Go to "Experiment Runner" Section
In the left control panel, scroll down to see:
- **n_estimators** (comma-separated): `100,200,500`
- **Train %** (comma-separated): `65,70`
- **Val %** (comma-separated): `15,20`
- **Early stopping** (comma-separated): `0,50`
- **Mode**: Select "grid" or "random"

### Step 2: Set Grid Search Mode
```
n_estimators: 100,200,500
train_pct: 65,70
val_pct: 15,20
early_stopping: 0,50
Mode: grid
```

Click "Run Experiments" ➜ **Tests 24 combinations**

### Step 3: Set Random Search Mode
```
n_estimators: 100,200,500
train_pct: 65,70
val_pct: 15,20
early_stopping: 0,50
Mode: random
Random trials: 10
```

Click "Run Experiments" ➜ **Tests 10 random combinations**

## What Gets Saved

Each experiment run creates:

```
experiments/exp_20251116_201413.csv

Columns:
  - n_estimators       (trees in model)
  - train_pct          (% of data for training)
  - val_pct            (% of data for validation)
  - early_stopping     (stopping rounds)
  - roc_auc            (model quality: 0.5 random, 1.0 perfect)
  - accuracy           (% correct predictions)
  - final_value        (portfolio end value, multiplier of $100k)
```

### How to Read Results

**Best Model** = highest `roc_auc` AND highest `final_value`

From the example file:
```
Row: 500,70,15,0,0.510,0.391,1.714
     ↑   ↑  ↑  ↑  ↑     ↑     ↑
     Trees %train %val stopping AUC Accuracy Return

This combination:
- 500 trees ✅
- 70% training ✅
- 15% validation ✅
- No early stopping ✅
- 0.510 AUC (best in this run!)
- 39% accuracy
- 1.714x return (+71.4%)
```

## Why Run Experiments?

### Find Optimal Parameters
Instead of guessing, systematically test combinations:
- "Does 500 trees work better than 200?"
- "Is 65% training data better than 70%?"
- "Should I use early stopping?"

### Avoid Overfitting
Grid search with proper validation reveals which parameters generalize best.

### Reproducibility
Every experiment is logged with all parameters, so you can:
- Reproduce exact results later
- Compare different configurations
- Document what you tried

## Example Workflow

### Scenario: Your default backtest gets 50% accuracy

**Question**: "Can I improve this?"

**Solution**: Run grid search with:
```
n_estimators: [200, 500, 1000]     ← Try different model sizes
train_pct: [60, 65, 70]            ← Try different splits
val_pct: [10, 15, 20]              ← Try different validation sizes
early_stopping: [0, 50, 100]       ← Try different stopping strategies
```

**Result**: CSV with 3×3×3×3 = 81 experiments

**Analysis**: Find row with best AUC + final_value

**Outcome**: "Use 500 trees, 70% training, 15% validation, 50 early stopping"

**Update**: Change your dashboard defaults to those values

## Current Experiments in Your Project

You have 2 experiment files:

### `exp_20251116_201413.csv` (26 rows)
```
24 grid search combinations
n_estimators: [100, 200, 500]
train_pct: [65, 70]
val_pct: [15, 20]
early_stopping: [0, 50]

Best result: n_est=500, train=70%, val=15%, es=0
ROC AUC: 0.510 (good!)
Final Value: 1.714x (+71.4% return)
```

### `exp_20251116_203905.csv` (another grid search)
Similar structure, different timestamp

## Pro Tips

### 1. Start Small
Don't grid 10 parameters with 10 options each (10 billion tests!).
Start with 3-4 key parameters, 2-3 options each.

### 2. Use Random Search for Big Spaces
If you have 1000+ combinations, use random with 50-100 trials instead.

### 3. Look at ROC AUC First
Don't just maximize `final_value` (lucky streak).
Look for **high ROC AUC** (model is actually good).

### 4. Check Accuracy Too
If accuracy drops significantly with more trees, might be overfitting.

### 5. Compare to Baseline
Make sure best grid result beats your current "run single backtest" settings.

## Where to Find Experiments

### In the Dashboard
After running experiments:
1. Results display in a **table on the dashboard**
2. Shows all parameters + results
3. Can see which combination was best

### In File System
```
📁 <repository_root>/
   └── experiments/
       ├── exp_20251116_201413.csv
       ├── exp_20251116_203905.csv
       └── exp_[YYYYMMDD_HHMMSS].csv  ← Each run creates a new file
```

### Open in Excel/Sheets
1. Download CSV from `experiments/`
2. Open in Excel or Google Sheets
3. Sort by `roc_auc` descending
4. Find best model

## Quick Commands

```bash
# See all experiments
dir experiments\

# See latest experiment
ls -lt experiments\ | head -1

# Open latest in Excel
start experiments\exp_20251116_201413.csv

# Count experiments
ls experiments\ | wc -l    # How many experiment files
```

## Troubleshooting

### Experiment Takes Forever
You have too many combinations!
```
Grid: 5×5×5×5 = 625 experiments (might take 30-60 minutes)
Solution: Use Random search with 50 trials instead
```

### CSV File Empty or Only Headers
Experiment failed. Check:
- Do you have enough data? (Try longer date range)
- Are your splits valid? (train + val < 100%)
- Any errors in logs?

### Results Look Random
That's normal! Slight variations in:
- Data splitting
- Model randomness (set seed if needed)
- Market regime changes

### All ROC AUC Scores ~0.5
Model is basically guessing!
- More/different features needed
- Different strategy entirely
- Market was choppy during period

## Summary

| Concept | Definition |
|---------|-----------|
| **Experiment** | One backtest with specific parameters |
| **Grid Search** | Test all combinations |
| **Random Search** | Test random sample of combinations |
| **Location** | `experiments/exp_YYYYMMDD_HHMMSS.csv` |
| **Best Result** | High AUC + High final_value |
| **Use Case** | Find optimal hyperparameters |

---

**Next Steps**: Try running a grid search with different parameters, compare results, and pick the best configuration! 🚀

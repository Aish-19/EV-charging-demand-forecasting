# Ablation Studies Plan — EV Demand Forecasting

**When to run:** After first working LSTM (around Milestone 5.5), BEFORE finalizing model choice.

---

## Project Context

| Setting | Value |
|---------|-------|
| Use case | Real-time stall availability (mobile app) |
| Cadence | 5 minutes |
| Lookback | 24 steps (2 hours) — DEFAULT |
| Horizon | 12 steps (1 hour ahead) |
| Target | `busy` (# stalls occupied) |

---

## Why Run Ablations?

Ablation studies prove each design decision earned its place. They convert:
- "I used LSTM with attention" → "I tested LSTM vs LSTM-ATT, and attention added X% improvement at Y% training cost"

This is the #1 differentiator in ML portfolio projects.

---

## Ablation 1: Lookback Window Size (PRIORITY 1)

**Hypothesis:** PACF suggests ~30 min of direct effect, but LSTM benefits from more context. Expect 2 hr sweet spot.

**Configurations to test:**

| Lookback (steps) | Time | Hypothesis |
|------------------|------|------------|
| 6 | 30 min | Strict PACF — minimum viable |
| 24 | 2 hours | My default — sweet spot? |
| 72 | 6 hours | Marginal gains? |
| 288 | 24 hours | Paper's choice — overkill? |

**Expected finding:** 24 steps performs similarly to 288 steps at 1/4 the training cost.

**Code template:**
```python
lookback_results = []
for lb in [6, 24, 72, 288]:
    model = train_lstm(lookback=lb, features=STANDARD_FEATURES)
    metrics = evaluate(model, val_loader)
    lookback_results.append({'lookback': lb, **metrics})

df = pd.DataFrame(lookback_results)
# Plot: lookback vs val_mse — find the knee
```

---

## Ablation 2: Feature Groups (PRIORITY 1)

**Hypothesis:** Each feature group should earn its place. Weather may not help if the daily pattern dominates.

**Configurations to test:**

| Config | Features included |
|--------|-------------------|
| Univariate | `busy` only |
| + Time | `busy`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` |
| + Weather | Above + `temperature`, `humidity`, `pressure`, `rainfall` |
| + Holiday | Above + `is_holiday` |
| All features | Above + `busy_smooth` (rolling mean) |

**Expected finding:** Time features add the most. Weather may add <1%. Holiday adds spikes during Golden Week / Spring Festival only.

---

## Ablation 3: Attention Mechanism (PRIORITY 2)

**Hypothesis:** Attention may not help for a simple daily pattern. Must prove it earns its place.

**Configurations to test:**

| Model | Description |
|-------|-------------|
| LSTM | Baseline — no attention |
| LSTM-ATT (additive) | Tanh + Softmax over timesteps |
| LSTM-ATT (dot-product) | Simpler attention variant |

**Expected finding:** For regular daily patterns, attention adds <1% improvement. Document honestly.

---

## Ablation 4: Hidden Size (PRIORITY 3)

**Hypothesis:** Diminishing returns beyond moderate size.

**Configurations to test:**

| Hidden size | Model params | Expected |
|-------------|--------------|----------|
| 32 | Smallest | Underfits? |
| 64 | Small | Good baseline |
| 96 | Medium (paper's choice) | Default |
| 128 | Large | Marginal gain? |
| 256 | Very large | Overfits? |

---

## Ablation 5: Cadence (PRIORITY 3 — optional)

**Hypothesis:** 5-min best for real-time app; coarser cadences are faster but less responsive.

**Configurations to test:**

| Cadence | Points/day | Trade-off |
|---------|------------|-----------|
| 5 min | 288 | Best responsiveness, noisiest |
| 10 min | 144 | Compromise |
| 15 min | 96 | Industry standard |

Note: This requires rebuilding the pipeline. Only run if time permits.

---

## Reusable Ablation Framework

```python
import pandas as pd
import time

def run_ablation(configs, train_fn, eval_fn, train_loader, val_loader):
    """Run multiple experiments and collect results."""
    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running config: {config}")
        start = time.time()
        model = train_fn(**config, train_loader=train_loader, val_loader=val_loader)
        train_time = time.time() - start
        metrics = eval_fn(model, val_loader)
        results.append({
            **config, 
            **metrics, 
            'train_time_sec': train_time,
            'n_params': sum(p.numel() for p in model.parameters())
        })
    return pd.DataFrame(results)
```

---

## Reporting Template for Each Ablation

For each ablation, produce:

1. **Table:** All configs with val_mse, train_time, params
2. **Plot:** Metric vs. varying parameter
3. **One-sentence finding:** "X performed best; Y added negligible improvement at Z% extra cost"

---

## Portfolio Impact

Example write-up:

> *"Through 4 ablation studies (lookback, features, attention, hidden size), I found:
> - 2-hour lookback matched 24-hour lookback at 1/4 training cost
> - Time features drove 85% of accuracy; weather added 0.8%
> - Attention added 1.2% MSE improvement — marginal
> - Hidden size 64-96 optimal; larger sizes overfit
>
> Final model: 2-hr lookback, time + holiday features, LSTM-96 without attention. Divergent from paper's design but better-suited to residential charging pattern."*

That one paragraph is worth more than 1,000 lines of code.

---

## Running Order When Time Comes

1. First working LSTM at default config
2. Ablation 1 (lookback) — fastest, most impactful
3. Ablation 2 (features) — most interpretable
4. Ablation 3 (attention) — key comparison vs paper
5. Ablation 4 (hidden size) — if time permits
6. Pick winning config based on validation
7. Re-train on train+val, evaluate on test (ONCE)

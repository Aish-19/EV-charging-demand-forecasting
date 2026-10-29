# EV Charging Demand Forecasting

Short-horizon forecasting of EV charging stall availability at a residential charging station,
using a mix of statistical baselines and deep-learning models in PyTorch.

---

## Goal

Predict the **number of occupied charging stalls** (`busy`) at a single residential EV station
over the **next hour**, in 5-minute steps. The intended end use is a real-time stall-availability
view: *"If I leave now, will there be a free stall when I arrive?"*

| Setting | Value |
|---------|-------|
| Target | `busy` — count of stalls currently in use (0–36) |
| Cadence | 5 minutes |
| Lookback (default) | 24 steps = 2 hours of history |
| Horizon | 12 steps = 1 hour ahead |
| Split | Chronological train / val / test |

The lookback and horizon are grounded in the data's own autocorrelation structure (see EDA
findings below), not copied from any reference.

---

## Dataset

A single CSV, `data/raw/ev_demand_history.csv`, holds one row every 5 minutes from
**2022-09-01 00:00** to **2023-02-28 23:55** — about 181 days, 52,128 rows.

Columns include `time`, `busy`, `idle`, `s_price`, `e_price`, `fast_busy`, `fast_idle`,
`slow_busy`, `slow_idle`, `duration`, and `volume`. Station capacity is a stable 36 stalls
across the whole period.

The hourly weather file in `data/raw/weather_central.csv` overlaps the demand date range and
will be joined by timestamp in the feature-engineering notebook.

---

## Repository Structure

```
Time series Pytorch/
├── README.md                          <- this file
├── data/
│   ├── raw/                           <- original source files (gitignored)
│   └── processed/
│       └── demand_clean.csv           <- output of EDA notebook
├── notebooks/
│   ├── 01_EDA.ipynb                   <- exploratory data analysis
│   └── 02_features_and_models.ipynb   <- (next) features + baselines + LSTM
├── docs/
│   └── ABLATION_STUDIES_PLAN.md       <- planned experiments
└── outputs/
    ├── figures/
    └── results/
```

---

## EDA Findings

The EDA notebook is self-contained. Each section follows a *heading → results → insight → next step*
pattern. The highlights:

### 1. Time axis is clean

Perfect 5-minute grid, zero duplicate timestamps, zero gaps in the time index, and station capacity
is stable at 36 stalls across all six months. No pre-processing needed on the time axis itself.

### 2. Missing values are sensor outages, not random dropouts

Of the 52,128 rows, 992 (~1.9%) are NaN in `busy`. These sit in **five clean blocks**:

| Block | Duration |
|-------|----------|
| 2022-09-14 to 2022-09-15 | ~26 hours |
| 2022-10-01 to 2022-10-02 | ~26 hours |
| 2022-10-24 (early morning) | ~4 hours |
| 2022-10-26 to 2022-10-27 | ~26 hours |
| 2022-10-27 (afternoon) | 30 minutes |

In every missing row, `busy`, `idle`, and `volume` are all NaN together — this is a
whole-row sensor dropout, consistent with a logger outage rather than a column bug.

**Handling:** hybrid rule — forward-fill (LOCF) gaps shorter than 6 hours, drop gaps 6 hours
or longer. Carrying a 3 AM value forward through a 26-hour hole would fabricate a flat line
through a full daily cycle; it's cleaner to drop those windows and ensure no LSTM window
spans them later.

### 3. Oct 21–24 is a sensor anomaly, not real demand

Visual inspection showed Oct 21–24 was unusually "jaggy." Three diagnostics confirmed it:

- Mean, std, max are in line with normal weekends — the amplitude isn't abnormal.
- Short-term volatility (average absolute change between consecutive 5-min readings) is
  **~2.57 vs ~0.2 for normal weekends** — roughly 10× the usual rate of change.
- Zoom-in shows large consecutive-point jumps that don't follow a daily shape.

The amplitude was fine; the jaggedness was the signal of sensor noise. Dropped the window
(~1,150 rows, ~2% of the data).

### 4. Residential overnight-charging pattern

Average `busy` by hour of day:

| Hour | Avg busy | | Hour | Avg busy |
|------|---------:|-|------|---------:|
| 00 | 12.7 | | 12 | 7.0 |
| 03 | 12.3 | | 14 | 6.4 |
| 06 | 11.8 | | 15 | 6.3 (lowest) |
| 09 | 7.5 | | 18 | 7.8 |
| 11 | 6.7 | | 22 | 12.3 |

Peak demand is **overnight (00:00–05:00)**, trough is **mid-afternoon (14:00–16:00)**.
Weekday and weekend averages are close — the daily shape dominates. No seasonal drift
across Sep → Feb.

This is a residential pattern: owners plug in when they get home in the evening, cars charge
through the night, then unplug and leave in the morning.

### 5. ACF spike at 24 h, but PACF cuts off at lag 2

Running ACF/PACF on 15-minute-resampled data (so lag 96 = 24 h):

- **ACF** is high for the first ~10 lags, dips to a minimum around lag 50 (12 h — opposite phase
  of the daily cycle), and climbs back to a **spike of ~0.7 at lag 96** (24 h).
- **PACF** has significant spikes only at **lag 1 and lag 2**; everything from lag 3 onward sits
  inside the 95% confidence band.

The reconciliation: the ACF spike at lag 96 is caused by a **common driver** (hour of day),
not by direct 24-hour dependence. The series doesn't "remember" yesterday — it follows a
schedule. The *only* raw-signal direct dependence is on the previous ~30 minutes (short-term
momentum, AR(2) signature).

**Modeling consequences:**

- Lookback does **not** need to be 24 hours of raw history. ~2 hours is a defensible default.
- The daily cycle must enter the model as an explicit **hour-of-day feature** (sin/cos).
- A seasonal-naive baseline (predict `busy_t` = `busy_{t-96}`) will be strong — the LSTM needs
  to beat it to justify its complexity.

---

## Decisions Carried Into Modeling

| Decision | Value | Why |
|----------|-------|-----|
| Target | `busy` | Direct answer to "will there be a stall?" |
| Cadence | 5 min (raw) | Real-time view needs fresh data |
| Lookback | 24 steps = 2 h (default) | PACF-informed, with buffer; ablation planned |
| Horizon | 12 steps = 1 h | Covers "driving home" scenario |
| Must-have features | hour sin/cos, day-of-week sin/cos | ACF spike at lag 96 is driven by hour |
| Weather features | temperature, humidity, pressure, rainfall | To be joined by timestamp in next notebook |
| Train/val/test | Chronological, never random | Standard time-series discipline |
| Baselines | Persistence, seasonal-naive, AR(2) | Fair comparison bar for the LSTM |

---

## Planned Experiments

See `docs/ABLATION_STUDIES_PLAN.md` for the full plan. Priority order:

1. **Lookback size** — {6, 24, 72, 288} steps. Test whether 2 hours really is enough.
2. **Feature groups** — univariate → +time → +weather → +holiday. Prove each group earns its place.
3. **Attention on/off** — does additive attention beat a plain LSTM here?
4. **Hidden size** — {32, 64, 96, 128}.

Each ablation produces a table (val MSE, train time, parameter count) and a plot (metric vs varying parameter).

---

## Status

- [x] EDA complete — `notebooks/01_EDA.ipynb`
- [x] Cleaned data saved to `data/processed/demand_clean.csv`
- [ ] Feature engineering + multi-source join (timestamp-based)
- [ ] Baselines (persistence, seasonal-naive, AR(2))
- [ ] Univariate LSTM
- [ ] Multivariate LSTM
- [ ] Attention mechanism
- [ ] Ablation studies
- [ ] Final evaluation + SHAP

---

## How to Reproduce

1. Clone the repo.
2. Place `ev_demand_history.csv` and `weather_central.csv` in `data/raw/`.
3. Open `notebooks/01_EDA.ipynb` in Colab (or JupyterLab), update the file path at the top,
   and run top to bottom. It produces `data/processed/demand_clean.csv`.
4. (Next) Open `notebooks/02_features_and_models.ipynb`.

Required packages: `numpy`, `pandas`, `matplotlib`, `statsmodels`, `openpyxl`,
`torch`, `scikit-learn`. All available via `pip`.

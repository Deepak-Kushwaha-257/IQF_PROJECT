# Deep Generative Modeling for Financial Time Series (VaR)
## Replication of Ericson et al. (2024) — arXiv:2401.10370

---

## Project Overview
This project replicates and extends the comparative study of **14 models** for
forecasting the conditional distribution of financial risk factor returns for
**Value at Risk (VaR)** estimation.

### Model Families
| Family | Models |
|--------|--------|
| Historical Simulation | PHS, FHS |
| Parametric | AR, AR-RET, GARCH, GARCH-RET, GARCHt-RET, NS-VS |
| Neural Networks | CGAN-FC, CGAN-LSTM, CWGAN, DIFFUSION, SIG, VAE |

---

## Folder Structure
```
iqf_project/
├── README.md
├── requirements.txt
├── setup.sh                      # One-click environment setup
├── run_experiment.py             # Master experiment runner
├── configs/
│   ├── data_config.yaml
│   ├── cgan_fc_config.yaml
│   ├── cwgan_config.yaml
│   ├── cgan_lstm_config.yaml
│   ├── vae_config.yaml
│   └── sig_config.yaml
├── data/
│   ├── raw/                      # Downloaded FRED CSVs
│   ├── processed/                # Windowed arrays (.npy)
│   └── simulated/                # GARCH/CIR simulated paths
├── models/
│   ├── base_model.py             # Abstract base class
│   ├── historical/
│   │   ├── phs.py
│   │   └── fhs.py
│   ├── parametric/
│   │   ├── ar_model.py
│   │   ├── garch_model.py
│   │   └── ns_vasicek.py
│   └── neural/
│       ├── cgan_fc.py
│       ├── cgan_lstm.py
│       ├── cwgan.py
│       ├── diffusion.py
│       ├── sig_model.py
│       └── vae_model.py
├── kpi/
│   ├── distribution_kpi.py
│   ├── acf_kpi.py
│   ├── backtesting_kpi.py
│   └── composite_score.py
├── utils/
│   ├── data_pipeline.py
│   ├── simulate_data.py
│   └── visualization.py
├── scripts/
│   ├── download_fred_data.py
│   └── run_all_models.sh
└── results/
    ├── tables/
    └── plots/
```

---

## Quick Start

### Step 1 — Clone & setup environment
```bash
git clone <your-repo>
cd iqf_project
bash setup.sh          # creates conda env, installs all packages
conda activate iqf_env
```

### Step 2 — Get FRED API key (free)
1. Go to https://fred.stlouisfed.org/docs/api/api_key.html
2. Create an account and request a free API key
3. Create `.env` file:
```
FRED_API_KEY=your_key_here
```

### Step 3 — Download data
```bash
python scripts/download_fred_data.py
```

### Step 4 — Run minimum viable experiment (PHS + GARCHt + CWGAN + CGAN-FC)
```bash
python run_experiment.py --models phs garch_t cwgan cgan_fc --dataset usdyc2
```

### Step 5 — Run full experiment (all 14 models)
```bash
python run_experiment.py --models all --dataset all
```

---

## Expected Results (USDYC1)
| Rank | Model | DIST | ACF | BT | Composite |
|------|-------|------|-----|----|-----------|
| 1 | PHS | 0.197 | 0.750 | 0.537 | 1.484 |
| 2 | GARCHt-RET | 0.734 | 0.789 | 0.545 | 2.068 |
| 3 | CWGAN | 0.766 | 0.764 | 0.565 | 2.094 |
| 4 | SIG | 0.807 | 0.876 | 0.551 | 2.234 |
| 5 | AR-RET | 0.971 | 0.706 | 0.572 | 2.248 |
| 6 | VAE | 0.997 | 0.931 | 0.556 | 2.484 |
| 7 | CGAN-FC | 0.991 | 0.814 | 0.806 | 2.611 |

**Lower composite score = better model.**

---

## Key Findings
- **PHS always wins** on short-horizon VaR for interest rates
- **GARCH (volatility clustering)** is a stronger baseline than plain AR
- **CWGAN** is the best neural network model
- More NN parameters ≠ better performance

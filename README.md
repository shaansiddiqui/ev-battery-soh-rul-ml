# Explainable EV Battery SOH Monitoring and RUL Analysis

Machine-learning project for predicting lithium-ion battery State of Health (SOH) and Remaining Useful Life (RUL) using the NASA PCoE Battery Aging Dataset.

## Key Features

- Cycle-level feature extraction from NASA PCoE Li-ion battery data
- SOH calculation from measured capacity
- Leave-One-Battery-Out validation across B0005, B0006, B0007, and B0018
- Linear Regression, Random Forest, and XGBoost model comparison
- XGBoost SOH model with mean LOBO R² = 0.850 and MAE = 3.21 percentage points
- SHAP global and local explainability
- Interactive Streamlit battery-health dashboard
- Exploratory RUL modelling and battery-specific degradation analysis

## Run

```powershell
& ".\.venv\Scripts\python.exe" "src\lobo_soh_validation.py"
& ".\.venv\Scripts\python.exe" "src\model_comparison_soh.py"

## Explainable Dashboard

Run the complete pipeline first:

```powershell
& ".\.venv\Scripts\python.exe" "src\model_comparison_soh.py"
& ".\.venv\Scripts\python.exe" "src\explain_soh_xgboost.py"

## RUL Finding

Direct global RUL prediction was evaluated using Linear Regression, Random Forest, and XGBoost. Results varied significantly between batteries because degradation rates were battery-specific. Therefore, the current validated deliverable focuses on SOH monitoring, while RUL forecasting remains an ongoing research component.

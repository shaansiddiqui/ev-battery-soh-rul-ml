# EV Battery SOH and RUL Prediction

Machine-learning project for predicting lithium-ion battery State of Health (SOH) and Remaining Useful Life (RUL) using the NASA PCoE Battery Aging Dataset.

## Features

- Cycle-level voltage, current, temperature, and discharge-duration features
- SOH calculation from measured capacity
- RUL calculation using 80% SOH as end-of-life threshold
- Leave-One-Battery-Out validation across B0005, B0006, B0007, and B0018
- Linear Regression, Random Forest, and XGBoost model comparison
- Planned SHAP explainability and Streamlit dashboard

## Run

```powershell
& ".\.venv\Scripts\python.exe" "src\lobo_soh_validation.py"
& ".\.venv\Scripts\python.exe" "src\model_comparison_soh.py"


### 5. Initialize Git

Run these commands in the project root:

```powershell
git init
git branch -M main
git status

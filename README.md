Telco Customer Churn Prediction Dashboard

## Problem
A telecom company wants to identify customers who are likely to cancel their
subscription (churn) so retention teams can act before it happens. This project
builds and compares several models that predict churn from customer account
and service data, then surfaces the main drivers behind each prediction.

**Users:** Customer retention / CRM team.
**Target:** `Churn` (binary — 1 = customer left, 0 = customer stayed).

## Data
- Source: Telco customer dataset (`telco_prep.csv`), 7,032 customers, 26 raw columns.
- Features cover demographics (gender, senior citizen, partner, dependents),
  account info (tenure, contract type, payment method, billing), and
  subscribed services (phone, internet, security, streaming, etc.).
- Two columns — `sentiment` and `feedback_length` — were derived from
  AI-generated customer feedback that was itself generated from the churn
  outcome. These were **removed before modeling** as target leakage: keeping
  them would let the model see a proxy for the answer it's supposed to predict.
- Class balance: **73.4% stayed / 26.6% churned** — a moderate imbalance,
  handled with `class_weight="balanced"` during training.

## Approach
1. **Cleaning** — standardized text fields, fixed data types, validated ranges
   (tenure 0–72, non-negative charges), removed duplicates and leakage columns.
2. **EDA** — explored churn rate by contract type, internet service, payment
   method, tenure, and customer segment (see Results summary for findings).
3. **Feature engineering** — encoded categoricals (one-hot), scaled numerics
   (fit on train only), added engineered flags (`is_month_to_month`,
   `is_fiber_optic`, `is_electronic_check`, `avg_monthly_spend`).
4. **Modeling** — three models trained and evaluated on an identical
   80/20 stratified train/test split (`random_state=42`):
   - Logistic Regression (`class_weight="balanced"`)
   - Random Forest (`n_estimators=300`, `class_weight="balanced"`)
   - ANN in Keras (64→32→1, dropout 0.3/0.2, `class_weight` balanced,
     EarlyStopping on `val_auc`)
5. **Clustering** — K-Means (k=4) on tenure, monthly charges, total charges,
   and total subscribed services, to find customer archetypes.
6. **Evaluation** — same metrics for every model (Accuracy, Precision, Recall,
   F1, ROC-AUC) on the same test set, plus confusion matrices and ROC curves.

See `results_model_summary.docx` for full metrics, drivers, and limitations.

## How to Run
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn joblib

# 1. Clean the raw data
python pipeline.py          # reads telco_prep.csv, writes telco_prep_cleaned.csv

# 2. Train classical models
python modeling.py          # Logistic Regression + Random Forest, saves artifacts.pkl

# 3. Train the ANN
python ann.py                # trains the Keras model on the same split

# 4. Run clustering
python clustering.py

# 5. Launch the demo UI
streamlit run app.py
```

## Project Structure
```
data/           telco_prep.csv, telco_prep_cleaned.csv
notebooks/       EDA, modeling, evaluation notebooks
models/          saved model artifacts (.pkl / .keras)
app/             Streamlit UI (app.py)
```

## Repository Layout Notes
- `final_model.pkl` or `final_model.keras` — the selected production model
  (see Results summary for which model was chosen and why).
- `scaler.pkl` — fitted `StandardScaler` for numeric features, needed to
  transform new input the same way training data was transformed.
- `kmeans_model.pkl` — fitted clustering model used by the UI to show which
  customer segment a prediction belongs to.

## Team
Built by Salma and Maya as a capstone project.
- **Data cleaning:** Maya
- **EDA:** Salma
- **Clustering, classical models, feature importance:** Maya
- **ANN, evaluation framework:** Salma
- **UI:** Maya (input) / Salma (output)
- **Documentation:** Maya (README, PM case study) / Salma (results summary)

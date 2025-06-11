# Supervised ML Assignment

This repository contains a modular and production-ready implementation of a supervised machine learning assignment, split into two tasks:

- **Classification Task:** Predict customer churn.
- **Regression Task:** Predict CO₂ emissions from car engine data.

---

## Project Structure

```
supervised_ml_assignment/
├── classification/
│   ├── data_loader.py    # Loads churn data
│   ├── preprocess.py     # Encoding and scaling
│   ├── model.py          # Random Forest Classifier
│   ├── evaluate.py       # Classification metrics
│   └── main.py           # Entry script
│
├── regression/
│   ├── data_loader.py    # Loads emissions data
│   ├── preprocess.py     # Feature scaling
│   ├── model.py          # Linear Regression
│   ├── evaluate.py       # Regression metrics
│   └── main.py           # Entry script
│
├── data/                 # CSV files (Telco-Customer-Churn.csv, CO2 Emissions_Canada.csv)
├── requirements.txt      # Required Python packages
└── README.md             # Project overview
```

---

## Tasks Implemented

### Classification: Customer Churn Prediction
- **Dataset:** `Telco-Customer-Churn.csv`
- **Preprocessing:**
  - Drop `customerID`
  - Convert `TotalCharges` to numeric
  - Label Encoding for categorical columns
  - Standard Scaling
- **Model:** `RandomForestClassifier`
- **Evaluation:**
  - Classification Report
  - Confusion Matrix
  - ROC Curve

Run:
```bash
python3 classification/main.py
```

---

### Regression: CO₂ Emissions Prediction
- **Dataset:** `CO2 Emissions_Canada.csv`
- **Features:**
  - Engine Size(L)
  - Cylinders
  - Fuel Consumption Comb (L/100 km)
- **Target:** CO2 Emissions(g/km)
- **Preprocessing:**
  - Standard Scaling
- **Model:** `LinearRegression`
- **Evaluation:**
  - MAE, MSE, RMSE, R² Score
  - Actual vs Predicted plot
  - Residual distribution plot

Run:
```bash
python3 regression/main.py
```

---

## Setup Instructions

1. Clone this repository
```bash
git clone https://github.com/rahulsharma9001/AI-ML-PiShaped.git
cd supervised_ml_assignment
```
2. Create and activate virtual environment (optional but recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```
3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Data Files Required
Place these CSVs inside the `data/` folder:
- `Telco-Customer-Churn.csv`
- `CO2 Emissions_Canada.csv`
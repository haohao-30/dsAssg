# Daily Gold Price H1 Prediction Application

This portable Streamlit prototype supports the BMDS2003 Daily Gold Price project. It compares four frozen regression models and produces an H1 prediction from information available at the latest completed market observation.

The application is an educational prototype, not financial advice. Successful model loading does not guarantee an accurate future gold-price forecast.

## H1 meaning

H1 means one **recorded observation** ahead:

```text
Latest completed observation at time t
        ↓
Predict Target_Next_Return at t+1
        ↓
Reconstruct the next recorded Gold Price
```

H1 is not necessarily the next calendar day because the source dataset is organised by recorded observations. H7 and H30 are outside this application's scope.

## Project structure

```text
Gold_Price_H1_App/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── Gold_Price_Final_ModelData_External_Return.csv
├── models/
│   ├── MLR/
│   ├── SVR/
│   ├── KNN/
│   └── RF/
└── comparison/
    ├── H1_Model_Comparison_Metrics.csv
    ├── H1_Model_Ranking.csv
    ├── H1_Combined_WalkForward_Predictions.csv
    ├── H1_Comparison_Configuration.json
    └── figures/
```

All runtime paths are derived from the folder containing `app.py`. The application does not require a personal computer path, a notebook workspace or a network connection after installation.

## Extract and install

Extract the supplied final ZIP first. Keep the internal `data`, `models` and `comparison` folders beside `app.py`; do not move individual joblibs or CSV files to different locations.

Python 3.12 is recommended. From inside the extracted `Gold_Price_H1_App` folder, create and activate an isolated environment, then install the pinned packages.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Launch the application from the project folder:

```bash
streamlit run app.py
```

The app performs strict startup validation before showing either mode. It stops with a clear message if a required file, dataset hash, predictor contract, model pipeline or historical alignment check fails.

## Exact predictor contract

The four deployment models require the following 16 numeric predictors in this exact order:

```text
Current_Price
Current_Open
Current_High
Current_Low
Current_Volume
Price_Lag1
Price_Lag2
Current_CHG
MA_7
MA_30
Volatility_7
Volatility_30
Momentum_7
Momentum_30
USD_Index_Return_Lag1
US10Y_Real_Yield_Change_Lag1
```

Missing, non-numeric and infinite values are rejected. Users enter all 16 values directly in the grouped application form. The app never requests or uses `Target_Next_Return`, `Target_Next_Price`, `Target_Date` or `Split` as predictors.

## Return-to-price reconstruction

Each model produces `Predicted_Next_Return`. The next recorded price is reconstructed using:

```text
Predicted_Next_Price = Current_Price × (1 + Predicted_Next_Return)
```

The app also reports the predicted price change, percentage return and direction for MLR, SVR, KNN and Random Forest.

## Application modes

### 1. H1 Future Prediction

This mode calls the four frozen deployment joblibs. All 16 predictors are prefilled from the latest packaged canonical observation and remain manually editable. `Origin Date` is retained only for display and record-keeping and is never passed to a model. The app contains no file-upload control.

The deployment models were trained on all 3,073 modelling rows. They are intended only for future-style application predictions and must not be used to recreate the historical evaluation period.

### 2. Historical Model Comparison

This mode reads only the saved leakage-safe walk-forward CSV files. It displays the frozen ranking and metrics, saved prediction rows, actual-versus-predicted prices, RMSE, MAE, MAPE/NRMSE, prediction-error distribution, directional comparison and skill-versus-persistence charts. Model and date filters change only the view, not the stored results.

The deployment joblibs are not called for historical metrics or charts. KNN ranked first descriptively, but KNN and SVR only slightly exceeded persistence on reconstructed-price RMSE. Return R² is negative for all four models, while the high price-level R² is influenced by adjacent-price persistence.

## Leakage-safety boundary

- Future mode uses all-data deployment models only for user-supplied future-style inputs.
- Historical mode uses saved 615-step walk-forward predictions only.
- The app does not train, retune or update an estimator.
- The app does not recreate historical predictions from deployment joblibs.
- No comparison or ensemble joblib is created.
- Users cannot upload or execute replacement model files.

## Scope and disclaimer

Only H1 is supported. No H7 or H30 forecast is implemented.

This software is for coursework demonstration and educational analysis. Gold-price forecasts are uncertain, may be materially wrong and should not be treated as investment advice or as a guarantee of future performance.

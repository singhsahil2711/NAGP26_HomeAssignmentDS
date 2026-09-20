# Customer Churn Prediction

## Objective

Identify customers who may churn using a Decision Tree classifier.
Provide an API returning a churn prediction and probability.

## Demo Video URL
https://nagarro-my.sharepoint.com/:v:/p/sahil_singh02/IQBiBHFXkz7zS4pkgOHD2RWnAUWyfuIRv0eUHgkQmv9dCsc?nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJPbmVEcml2ZUZvckJ1c2luZXNzIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXciLCJyZWZlcnJhbFZpZXciOiJNeUZpbGVzTGlua0NvcHkifX0&e=REjBVl

## Project Files

- `TelcoCustomerChurn.csv`: supplied dataset.
- `TelcoCustomerChurn - Data Dictionary.csv`: column descriptions.
- `notebook/churn_analysis.ipynb`: analysis, modelling, and explanations.
- `model/churn_model.pkl`: fitted preprocessing and Decision Tree pipeline.
- `app.py`: FastAPI application.
- `requirements.txt`: recorded package versions.
- `sample_request.json`: example customer input.

## Environment Setup

Developed using Python 3.13.2.

From PowerShell in the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the Notebook

```powershell
.\.venv\Scripts\python.exe -m notebook
```

Open `notebook/churn_analysis.ipynb` and run its cells in order.

The notebook covers data inspection, cleaning, six EDA visualizations,
feature engineering, model comparison, evaluation, interpretation,
and model saving.

The final saving cell generates `model/churn_model.pkl`.
The API can use the included saved pipeline without rerunning training.

## Modelling Approach

- Target: `Churn`, encoded as No = 0 and Yes = 1.
- Exclude `customerID` from predictors.
- Convert TotalCharges to numerical values.
- Retain its 11 blank values as missing until pipeline imputation.
- Use a stratified 70:30 training/testing split with random_state=42.
- Perform predictive EDA using training data.
- Engineer AdditionalServiceCount and UsesAutomaticPayment.
- Apply median imputation and one-hot encoding inside the pipeline.
- Compare two Decision Tree configurations using three-fold stratified
  cross-validation within the training portion.
- Fit preprocessing separately within each cross-validation fold.

The selected tree uses max_depth=5, min_samples_leaf=20,
and random_state=42. Compared with the default tree, it achieved
higher validation precision and F1 with a small recall reduction.

The selected pipeline was refitted on all training data and evaluated
on the reserved testing portion.

## Final Evaluation

| Metric | Result |
|---|---|
| Accuracy | 79.41% |
| Precision | 61.45% |
| Recall | 60.25% |
| F1 | 0.6085 |

Confusion-matrix counts:
- True negatives: 1,340.
- False positives: 212.
- False negatives: 223.
- True positives: 338.

The model detected 338 of 561 churners and missed 223.
Of 550 customers flagged for outreach, 338 actually churned.

Recall is the primary business concern because missed churners represent
missed retention opportunities. Precision remains relevant because
incorrect flags consume outreach resources.

## Start the API

From the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs for interactive API documentation.

The saved pipeline loads once per application process. The API applies
the same numerical conversion and feature formulas as the notebook
before calling the fitted pipeline.

## Submit a Prediction Request

Endpoint: `POST /predict`

The sample JSON is provided in `sample_request.json`.

In the documentation page, expand POST /predict, select Try it out,
paste the sample JSON, and select Execute.

Alternatively, from a second PowerShell terminal in the project root:

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -InFile ".\sample_request.json"
```

Actual response obtained for the included sample:

```json
{
  "prediction": "Yes",
  "churn_probability": 0.543010752688172
}
```

The probability refers to Churn = Yes.

## Input Handling

Provide the 19 original predictor fields using the dictionary's
column names and category values. `customerID` is optional and ignored
for prediction.

Do not send Churn or engineered features; the API calculates the
engineered features.

TotalCharges must be present but may be JSON null. The saved imputer
handles that missing value. Other required fields must have valid values.

Missing required fields, incorrect types, invalid categories, negative
numerical values, non-finite charges, unexpected fields, and contradictory
service combinations are rejected with HTTP 422.

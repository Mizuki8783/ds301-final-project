# Credit Card Default Prediction

A machine learning project that predicts credit card default risk using the UCI Credit Card dataset. This project implements a Decision Tree classifier with comprehensive data preprocessing and feature engineering.

## 📊 Project Overview

This project analyzes credit card customer data to predict the likelihood of default payment in the next month. The model uses demographic information, credit history, payment status, and billing information to make predictions.

### Key Features
- **Dataset**: UCI Credit Card Dataset with 30,000 records and 24 features
- **Model**: Decision Tree Classifier optimized with GridSearchCV
- **Performance**: 
  - Accuracy: 78.1%
  - Precision: 50.4%
  - Recall: 55.0%
  - F1-Score: 52.6%
  - ROC-AUC: 74.8%

### Data Processing Pipeline
1. **Data Inspection**: Examined dataset structure, data types, and distributions
2. **Missing Value Handling**: Verified no missing values present
3. **Outlier Treatment**: Applied IQR method and capped extreme values at 1st/99th percentiles
4. **Categorical Variable Cleaning**: 
   - Recoded EDUCATION values (0, 5, 6 → 4 for "others")
   - Recoded MARRIAGE values (0 → 3 for "others")
   - Validated SEX values (1: Male, 2: Female)
5. **Feature Engineering**: Created GENDER_MARRIAGE combined feature (6 categories)
6. **Data Filtering**: Excluded divorced women records (232 samples)
7. **Train-Test Split**: 70/30 split with stratification (22.1% default rate maintained)

### Model Development
- **Algorithm**: Decision Tree Classifier
- **Hyperparameter Tuning**: GridSearchCV with 10-fold cross-validation
- **Optimization Target**: F1-score (to balance precision and recall)
- **Best Parameters**:
  - max_depth: 5
  - min_samples_split: 2
  - min_samples_leaf: 1
  - criterion: entropy
  - class_weight: {0: 1, 1: 3}

## 📁 Project Structure

```
ds301-final-project/
│
├── data/
│   └── UCI_Credit_Card.csv                # Primary dataset
├── models/                                # Saved models (joblib/pickle)
│   ├── decision_tree_model.pkl            # Tuned Decision Tree (UCI)
│   ├── xgboost_model.pkl                  # Tuned XGBoost (UCI)
│   ├── lightgbm_model.pkl                 # Tuned LightGBM (UCI)
│   ├── logistic_regression_model.pkl      # HMEQ experiment
│   └── svm_model.pkl                      # HMEQ experiment
├── notebooks/
│   ├── 01_create_models.ipynb             # UCI preprocessing + Decision Tree vs LightGBM
│   ├── 02_create_models_with_similar_dataset.ipynb  # HMEQ dataset (not included) + classical models
│   ├── 03_XGBoost.ipynb                   # Tuned XGBoost on UCI dataset
│   └── 04_lightGBM.ipynb                  # LightGBM deep dive + comparison tables
├── presentation.md                        # Slide deck content
├── pyproject.toml                         # Python 3.12 dependencies (pandas, scikit-learn, xgboost, lightgbm)
├── uv.lock
└── README.md
```

### Directory Details

- **`data/`**: Contains the UCI Credit Card dataset in CSV format
- **`models/`**: Stores the trained machine learning model as a pickle file
- **`notebooks/`**: Contains jupyter noterbooks where we processed the data and created various models

## 🚀 How to Run the Code

### Prerequisites

Ensure you have the following installed:
- Python 3.12 or higher
- Jupyter Notebook or JupyterLab
- Required packages (see below)

### Installation

1. **Clone the repository** (if applicable):
```bash
cd ds301-final-project
```

1. **Install dependencies** (Python 3.12):
   - With `uv` (recommended): `uv sync`
   - Or with pip: `pip install pandas numpy scikit-learn xgboost lightgbm ipykernel`
2. **Launch notebooks:** `uv run jupyter lab` (or `jupyter notebook`) and open any file in `notebooks/`.
3. **Data prep notes (UCI):**
   - No missing values; outliers capped at 1st/99th percentiles
   - EDUCATION recoded (0/5/6 → 4), MARRIAGE recoded (0 → 3), SEX validated
   - Engineered `GENDER_MARRIAGE`; excluded divorced women (232 rows)
   - Stratified 70/30 train-test split preserves the 22.1% default rate
4. **Optional HMEQ run:** Place `hmeq.csv` in the project root to execute `02_create_models_with_similar_dataset.ipynb`.

### Loading a Saved Model
```python
import joblib
model = joblib.load("models/lightgbm_model.pkl")
# preds = model.predict(X_new)
# probs = model.predict_proba(X_new)[:, 1]
```


## 📊 Dataset Information

### Features (24 total)
- **LIMIT_BAL**: Credit limit
- **SEX**: Gender (1=male, 2=female)
- **EDUCATION**: Education level (1=graduate, 2=university, 3=high school, 4=others)
- **MARRIAGE**: Marital status (1=married, 2=single, 3=others)
- **AGE**: Age in years
- **PAY_0 to PAY_6**: Repayment status for past 6 months
- **BILL_AMT1 to BILL_AMT6**: Bill statement amounts for past 6 months
- **PAY_AMT1 to PAY_AMT6**: Previous payment amounts for past 6 months
- **GENDER_MARRIAGE**: Engineered feature combining gender and marriage status

### Target Variable
- **default.payment.next.month**: Binary (1=default, 0=no default)
- Default rate: 22.1% of records

## 🎯 Model Performance (all notebooks)
**UCI Credit Card dataset** — `notebooks/01_create_models.ipynb`, `03_XGBoost.ipynb`, `04_lightGBM.ipynb`
| Model          | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|----------------|----------|-----------|--------|----------|---------|
| Decision Tree  | 0.781    | 0.504     | 0.550  | 0.526    | 0.748   |
| XGBoost        | 0.796    | 0.537     | 0.560  | 0.548    | 0.778   |
| LightGBM       | 0.789    | 0.521     | 0.582  | 0.550    | 0.779   |

LightGBM offers the best recall and F1 on UCI, while XGBoost edges slightly on accuracy.

**HMEQ dataset experiment** — `notebooks/02_create_models_with_similar_dataset.ipynb` (requires `hmeq.csv`)
| Model                  | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|------------------------|----------|-----------|--------|----------|---------|
| Decision Tree          | 0.876    | 0.689     | 0.689  | 0.689    | 0.790   |
| Logistic Regression    | 0.872    | 0.735     | 0.559  | 0.635    | 0.881   |
| Support Vector Machine | 0.902    | 0.838     | 0.630  | 0.719    | 0.912   |

## Presentation
Check out `Presentation Slides.pdf` or visit [this link](https://www.canva.com/design/DAG4mQSIvQc/IniMlac3Aq5eBG27yW9t9Q/edit?utm_content=DAG4mQSIvQc&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## Contributors
- Júlia Martins Santana Figueiredo
- Yuki Okabe
- Hajime Imaizumi
- Mizuki Nakano

## License
Uses the UCI Credit Card dataset for research purposes. Refer to the dataset terms for usage details.

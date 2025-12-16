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
ds301-midterm/
│
├── data/
│   └── UCI_Credit_Card.csv          # Raw dataset (30,000 records)
│
├── models/
│   └── best_decision_tree_model.pkl # Trained model (saved via joblib)
│
├── create_model.ipynb                # Main notebook with full ML pipeline
└── README.md                         # This file
```

### Directory Details

- **`data/`**: Contains the UCI Credit Card dataset in CSV format
- **`models/`**: Stores the trained machine learning model as a pickle file
- **`create_model.ipynb`**: The main Jupyter notebook containing all code for data processing, model training, and evaluation

## 🚀 How to Run the Code

### Prerequisites

Ensure you have the following installed:
- Python 3.12 or higher
- Jupyter Notebook or JupyterLab
- Required packages (see below)

### Installation

1. **Clone the repository** (if applicable):
```bash
cd ds301-midterm
```

2. **Install dependencies**:

If using `uv` (recommended):
```bash
uv sync
```

Or install packages manually:
```bash
pip install pandas numpy scikit-learn joblib jupyter
```

### Running the Notebook

1. **Start Jupyter Notebook**:
```bash
jupyter notebook
```

2. **Open the notebook**:
   - Navigate to `create_model.ipynb` in the Jupyter interface
   - Click to open

3. **Run all cells**:
   - Option 1: Click `Cell` → `Run All` in the menu
   - Option 2: Run cells sequentially using `Shift + Enter`

### Expected Outputs

The notebook will:
1. Load and inspect the dataset
2. Clean and preprocess the data
3. Engineer new features
4. Split data into training and test sets
5. Train a Decision Tree model with hyperparameter tuning (this may take several minutes)
6. Display model performance metrics
7. Save the trained model to `models/best_decision_tree_model.pkl`

### Using the Trained Model

To load and use the saved model:

```python
import joblib
import pandas as pd

# Load the model
model = joblib.load('models/best_decision_tree_model.pkl')

# Make predictions on new data
# predictions = model.predict(X_new)
# probabilities = model.predict_proba(X_new)
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

## 🎯 Model Performance

### Confusion Matrix
```
                 Predicted
                No Default  Default
Actual
No Default        5,885     1,069
Default             890     1,087
```

### Classification Report
```
              Precision  Recall  F1-Score  Support
No Default       0.87     0.85     0.86     6,954
Default          0.50     0.55     0.53     1,977

Accuracy                           0.78     8,931
```

## 🎨 Presentation

View the full project presentation here:
**[Credit Card Default Prediction - Canva Presentation](https://www.canva.com/design/DAG4mQSIvQc/IniMlac3Aq5eBG27yW9t9Q/edit)**


## 👥 Contributors

- Júlia Martins Santana Figueiredo
- Yuki Okabe
- Hajime Imaizumi
- Mizuki Nakano

## 📄 License

This project uses the UCI Credit Card dataset, which is publicly available for research purposes.

---

*For questions or issues, please refer to the presentation or contact the project team.*

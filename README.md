# 🏠 House Price Prediction using Machine Learning

An end-to-end machine learning regression project for predicting
residential house sale prices from structural, locational, quality, and
property-condition features.

## 📌 Project Overview

This project develops a complete regression workflow to predict **house
SalePrice** using historical residential property data.

The project covers:

-   Data understanding and quality auditing
-   Exploratory Data Analysis (EDA)
-   Univariate, Bivariate, and Multivariate (UBM) analysis
-   Hypothesis testing
-   Feature engineering
-   Missing-value handling
-   Categorical encoding
-   Leakage-aware preprocessing
-   Target transformation
-   Multiple regression model comparison
-   Cross-validation
-   Hyperparameter tuning
-   Residual and price-tier error analysis
-   SHAP-based model interpretability
-   End-to-end production pipeline
-   Model serialization and deployment validation

## 🎯 Objective

The main objective is to build a reliable regression model that
estimates residential house sale prices from available property
characteristics.

The target variable is:

``` text
SalePrice
```

The dataset contains **1,460 residential property records and 81 initial
features** covering land characteristics, zoning, property size,
construction quality, condition, basement and garage information, room
counts, and sale-related attributes.

## 📊 Dataset

The project uses a residential house-price dataset containing features
such as:

-   Overall quality and condition
-   Living area
-   Basement area
-   Garage capacity and area
-   Lot area
-   Number of rooms
-   Neighborhood
-   Year built
-   Remodeling information
-   Porch and outdoor-area measurements
-   Sale price

The target variable, `SalePrice`, is continuous, making this a
**regression problem**.

## 🔍 Project Workflow

### 1. Data Understanding & Quality Audit

The dataset was audited for:

-   Shape and dimensions
-   Data types
-   Missing values
-   Duplicate records
-   Numerical distributions
-   Categorical variables
-   Potential outliers

### 2. Exploratory Data Analysis

EDA follows the **Univariate, Bivariate, and Multivariate (UBM)**
framework.

The analysis includes:

-   SalePrice distribution
-   Numerical feature distributions
-   Categorical feature analysis
-   Price variation across neighborhoods
-   Relationship between important numerical features and SalePrice
-   Outlier inspection
-   Comprehensive correlation analysis across numerical features

The target distribution is right-skewed, with recorded skewness of
approximately **1.88**.

### 3. Hypothesis Testing

Three statistical relationships were tested using appropriate
statistical methods, including Pearson correlation and One-Way ANOVA.

The analysis supports statistically significant relationships between
house price and important factors such as:

-   Overall Quality
-   Neighborhood
-   Ground Living Area

For example, the Pearson correlation between `GrLivArea` and `SalePrice`
was approximately **0.7086**, with a highly significant p-value.

### 4. Feature Engineering

Domain-driven features were created to provide more meaningful
information to the models:

  Feature          Description
  ---------------- ------------------------------------------------------
  `TotalSF`        Combined usable square footage
  `TotalBath`      Combined bathroom measure
  `HouseAge`       Property age at sale
  `RemodAge`       Age since remodeling
  `IsRemodeled`    Indicates whether the property was remodeled
  `TotalPorchSF`   Combined porch/outdoor area
  `Qual_x_SF`      Interaction between quality and total square footage

These engineered variables help the models capture property size, age,
remodeling, outdoor space, and quality-size interactions.

### 5. Preprocessing

The preprocessing workflow handles different feature types
appropriately.

**Numerical features:** - Median imputation - Scaling where required for
linear models

**Categorical features:** - Most-frequent-value imputation - One-hot
encoding - `handle_unknown='ignore'` for safer inference

The final deployment workflow packages feature engineering and
preprocessing together to reduce training-serving mismatch risk.

### 6. Target Transformation

Because `SalePrice` is right-skewed, a logarithmic transformation is
used for the applicable models.

The workflow uses:

``` text
log1p(SalePrice)
```

and converts predictions back to the original dollar scale using:

``` text
expm1()
```

This is implemented through `TransformedTargetRegressor` in the relevant
model workflow.

## 🤖 Models Evaluated

The project compares multiple regression approaches:

-   Linear Regression (OLS)
-   Ridge Regression
-   Decision Tree Regressor
-   Random Forest Regressor
-   Gradient Boosting Regressor
-   XGBoost Regressor

Cross-validation and hyperparameter tuning were used to compare and
improve model performance.

## 📈 Model Performance

The final reported comparison is:

  --------------------------------------------------------------------------------------------------------------------------
  Model               CV R²           CV RMSE            CV MAE      Test R²         Test RMSE          Test MAE   Test MAPE
  ------------ ------------ ----------------- ----------------- ------------ ----------------- ----------------- -----------
  Linear                N/A               N/A               N/A       0.4539       \$64,723.22       \$19,686.76         N/A
  Regression                                                                                                     
  (OLS)                                                                                                          

  Ridge              0.7000       \$37,891.68       \$16,639.56       0.9069       \$26,719.27       \$16,325.74       9.67%
  Regression                                                                                                     

  Decision           0.8231       \$33,142.84       \$22,003.71       0.8725       \$31,266.39       \$22,366.94         N/A
  Tree                                                                                                           

  Random             0.8511       \$29,600.36       \$17,792.65       0.8909       \$28,921.90       \$17,420.58         N/A
  Forest                                                                                                         

  Gradient           0.8731       \$27,183.66       \$15,588.43       0.9015       \$27,480.76       \$16,165.78         N/A
  Boosting                                                                                                       

  **XGBoost      **0.8896**   **\$25,471.64**   **\$14,938.72**   **0.9049**   **\$27,010.63**   **\$15,463.37**   **8.99%**
  (Tuned)**                                                                                                      
  --------------------------------------------------------------------------------------------------------------------------

> **Note:** OLS cross-validation metrics were numerically unstable
> because of strong multicollinearity in the unregularized linear
> feature space. They are therefore reported as N/A rather than being
> interpreted.

### 🏆 Final Model

**Tuned XGBoost Regressor** is the recommended production candidate.

### Final XGBoost Metrics

-   **Mean CV R²:** 0.8896 ± 0.0324
-   **CV RMSE:** \$25,471.64
-   **CV MAE:** \$14,938.72
-   **Test R²:** 0.9049
-   **Test RMSE:** \$27,010.63
-   **Test MAE:** \$15,463.37
-   **Test MAPE:** 8.99%

### Why XGBoost?

XGBoost was selected primarily based on its **cross-validation
performance**, with the held-out test set used for final confirmation.

It provides:

-   Strongest reported CV R²
-   Lowest stable CV RMSE
-   Lowest stable CV MAE
-   Lowest Test MAE among the compared models
-   Lowest reported Test MAPE
-   Strong Test R²
-   Ability to capture non-linear relationships and feature interactions

Ridge achieved slightly better Test R² and Test RMSE, so XGBoost is
**not claimed to be best on every individual metric**. Instead, it
provides the strongest overall balance of cross-validation performance
and practical prediction error.

## 📉 Error & Diagnostic Analysis

The project includes:

-   Actual vs. Predicted analysis
-   Residual distribution analysis
-   Residual diagnostics
-   Price-tier error analysis

Prediction errors are not necessarily uniform across all properties.
Higher-value properties can produce larger absolute dollar errors
because the underlying price scale is larger.

The model should therefore be considered a **data-driven valuation
aid**, not an exact replacement for professional property appraisal.

## 🔬 Model Interpretability

Feature importance and SHAP analysis are used to understand how the
final XGBoost model uses the available features.

Important predictive factors include engineered and raw property
characteristics such as:

-   `Qual_x_SF`
-   `TotalSF`
-   `OverallQual`
-   `OverallCond`
-   `TotalBath`
-   `GarageCars`
-   `HouseAge`
-   `RemodAge`

> Feature importance and SHAP values describe **predictive
> association**, not causal relationships.

## 🚀 Deployment

The project includes an end-to-end production workflow:

``` text
Raw House Features
        ↓
Feature Engineering
        ↓
Preprocessing
        ↓
Tuned XGBoost Model
        ↓
Predicted Sale Price
```

The final pipeline is serialized as:

``` text
house_price_prediction_pipeline.joblib
```

The supporting production utility module is:

``` text
house_pipeline_utils.py
```

The notebook also validates the saved and reloaded pipeline using
deployment-style inputs and checks prediction consistency.

### ⚠️ Important Deployment Dependency

`house_pipeline_utils.py` must remain available with the
notebook/pipeline when the project is moved to another environment.

If the notebook cannot locate this module, Python may raise:

``` text
ModuleNotFoundError: No module named 'house_pipeline_utils'
```

Keep the notebook and utility module in the same project directory or
configure the Python path correctly.

## 📁 Recommended Repository Structure

``` text
House-Price-Prediction/
│
├── README.md
├── House_Price_Prediction_Final_Corrected.ipynb
├── house_pipeline_utils.py
├── house_price_prediction_pipeline.joblib
│
├── reports/
│   ├── House_Price_Model_Comparison_Performance_Recommendation_Report.docx
│   └── House_Price_Challenges_Faced_and_How_Handled_Report.docx
│
└── data/
    └── data.csv
```

> If the dataset is not permitted to be publicly redistributed, do not
> upload the raw dataset to a public GitHub repository. Instead, provide
> instructions for obtaining it or keep it outside the repository.

## 🛠️ Technologies Used

-   Python
-   Pandas
-   NumPy
-   Matplotlib
-   Seaborn
-   Scikit-learn
-   XGBoost
-   SHAP
-   Joblib
-   Jupyter Notebook

## ▶️ How to Run

### 1. Clone the repository

``` bash
git clone <your-github-repository-url>
cd House-Price-Prediction
```

### 2. Install dependencies

``` bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap joblib jupyter
```

Or, if a `requirements.txt` file is provided:

``` bash
pip install -r requirements.txt
```

### 3. Open the notebook

``` bash
jupyter notebook House_Price_Prediction_Final_Corrected.ipynb
```

Run the notebook cells sequentially.

### 4. Deployment

Ensure the following files are available in the project directory:

``` text
house_pipeline_utils.py
house_price_prediction_pipeline.joblib
```

The serialized pipeline can then be loaded for inference using `joblib`.

## ⚠️ Limitations

-   The model is trained on historical residential property data and may
    not represent future market conditions.
-   Prediction accuracy can vary across property price ranges.
-   Higher-value properties can have larger absolute dollar errors.
-   The model captures statistical patterns rather than causal
    relationships.
-   Production deployment should include monitoring for data drift and
    prediction performance.
-   Predictions should not be treated as guaranteed property valuations.

## 🔮 Future Improvements

Potential future enhancements include:

-   Monitoring feature and prediction drift
-   Periodic model retraining
-   Larger and more recent housing datasets
-   Additional location-specific features
-   More systematic model calibration and validation
-   API or web-based prediction interface
-   Automated production monitoring
-   Additional interpretability and fairness checks

## 📄 Project Reports

The project documentation includes:

-   **Model Comparison, Performance & Production Recommendation Report**
-   **Challenges Faced & How They Were Handled Report**

These reports document the model-selection reasoning, performance
results, production recommendation, technical challenges, and mitigation
strategies.

## 👤 Author

**Hemaa Shri**

B.Tech -- Artificial Intelligence and Data Science

**Project:** PRCP-1020\
**Project ID:** PTID-CDS-JUL-26-11212

## 📌 Project Status

**Status:** Completed

**Final Production Candidate:** Tuned XGBoost Regressor

**Test R²:** 0.9049

**Test MAE:** \$15,463.37

**Test MAPE:** 8.99%

**Deployment:** End-to-end pipeline serialization and reload validation
completed.

------------------------------------------------------------------------

⭐ If you find this project useful, consider giving the repository a
star!

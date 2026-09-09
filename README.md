# 🌾 Smart Agriculture: Crop Yield Prediction

A machine learning project for predicting **crop yield in tons per hectare** using agricultural, environmental, and farming-related features.

The project explores crop-yield patterns through exploratory data analysis and compares multiple regression algorithms — **Linear Regression, Random Forest, and Gradient Boosting** — to identify an effective predictive model.

The final Random Forest pipeline is serialized with Joblib for reuse.



## 📌 Project Overview

Accurate crop-yield prediction can support smarter agricultural planning by providing estimates of expected production based on factors such as:

* Rainfall
* Temperature
* Soil type
* Crop type
* Region
* Weather conditions
* Fertilizer usage
* Irrigation usage
* Days to harvest

This project treats crop yield as a **supervised regression problem**, with:

```text
Target:
Yield_tons_per_hectare
```

The workflow covers the complete machine learning pipeline:

```text
Raw Dataset
     ↓
Data Exploration
     ↓
Missing Value Analysis
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Train/Test Split
     ↓
Preprocessing
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Model Comparison
     ↓
Feature Importance
     ↓
Model Serialization
     ↓
Sample Prediction
```



## 🎯 Objectives

The primary objectives of this project are to:

* Explore agricultural factors associated with crop yield.
* Identify relationships between environmental variables and yield.
* Prepare categorical and numerical features for machine learning.
* Compare different regression algorithms.
* Evaluate models using MAE, RMSE, and R².
* Identify important features influencing predictions.
* Save the trained model for future inference.
* Demonstrate prediction on a new sample of agricultural conditions.



## 📊 Dataset

The project uses the `crop_yield.csv` dataset.

The notebook loads the dataset from:

```python
pd.read_csv(
    "/kaggle/input/datasets/samuelotiattakorah/agriculture-crop-yield/crop_yield.csv"
)
```

The dataset contains agricultural and environmental variables used to predict crop yield.

### Features

| Feature               | Type              | Description                    |
|  | -- |  |
| `Region`              | Categorical       | Geographic region              |
| `Soil_Type`           | Categorical       | Type of soil                   |
| `Crop`                | Categorical       | Crop being cultivated          |
| `Rainfall_mm`         | Numerical         | Rainfall amount in millimetres |
| `Temperature_Celsius` | Numerical         | Temperature in degrees Celsius |
| `Fertilizer_Used`     | Boolean → Integer | Whether fertilizer was used    |
| `Irrigation_Used`     | Boolean → Integer | Whether irrigation was used    |
| `Weather_Condition`   | Categorical       | Weather condition              |
| `Days_to_Harvest`     | Numerical         | Number of days until harvest   |

### Target

```text
Yield_tons_per_hectare
```

The target represents crop yield measured in tons per hectare.



## 🔎 Exploratory Data Analysis

The notebook performs several exploratory analyses before model training.

### Dataset Inspection

The dataset is inspected using:

* Dataset shape
* Column names
* Data types
* Dataset information
* Descriptive statistics

### Missing Values

Missing values are checked for every column and visualized using a heatmap.

The modeling pipeline also includes explicit imputation strategies.

### Yield Distribution

The distribution of:

```text
Yield_tons_per_hectare
```

is visualized using a histogram with KDE.

### Rainfall vs Yield

A scatter plot is used to investigate the relationship between:

```text
Rainfall_mm
```

and:

```text
Yield_tons_per_hectare
```

### Temperature vs Yield

The project also examines:

```text
Temperature_Celsius
```

against crop yield.

### Crop-wise Yield

A box plot compares yield distributions across different crop types.

These visualizations provide an initial understanding of the dataset before predictive modeling.



## 🧹 Feature Engineering

The notebook converts the boolean farming-practice variables into integers:

```python
df['Fertilizer_Used'] = df['Fertilizer_Used'].astype(int)
df['Irrigation_Used'] = df['Irrigation_Used'].astype(int)
```

This transforms the values into numerical representations suitable for the machine learning pipeline.

For example:

```text
False → 0
True  → 1
```



## 🧩 Feature & Target Definition

The target variable is separated from the predictor variables:

```python
X = df.drop('Yield_tons_per_hectare', axis=1)
y = df['Yield_tons_per_hectare']
```

### Numerical Features

```text
Rainfall_mm
Temperature_Celsius
Fertilizer_Used
Irrigation_Used
Days_to_Harvest
```

### Categorical Features

```text
Region
Soil_Type
Crop
Weather_Condition
```



## ⚙️ Data Preprocessing

A `ColumnTransformer` is used to apply different preprocessing operations to numerical and categorical variables.

### Numerical Pipeline

Numerical features go through:

```text
Missing-value imputation
        ↓
Median replacement
        ↓
StandardScaler
```

Implemented using:

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
```

### Categorical Pipeline

Categorical variables go through:

```text
Missing-value imputation
        ↓
Most-frequent replacement
        ↓
One-hot encoding
```

Implemented using:

```python
Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
```

Using:

```python
handle_unknown='ignore'
```

allows the pipeline to handle previously unseen categorical values during inference without failing.



## ✂️ Train/Test Split

The dataset is divided into:

```text
80% Training
20% Testing
```

using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The fixed random state ensures reproducibility of the split.



# 🤖 Machine Learning Models

Three regression algorithms are evaluated.



## 1. Linear Regression

The first baseline model is standard Linear Regression:

```python
LinearRegression()
```

The model is placed inside the preprocessing pipeline so preprocessing and prediction occur consistently.

Linear Regression provides a simple baseline against which the more complex ensemble models can be compared.



## 2. Random Forest Regressor

The second model is a Random Forest:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

Configuration:

| Parameter       | Value |
|  | -: |
| Number of trees |   100 |
| Random state    |    42 |
| Parallel jobs   |    -1 |

Random Forest is particularly useful for capturing nonlinear relationships and interactions between agricultural variables.



## 3. Gradient Boosting Regressor

The third model is:

```python
GradientBoostingRegressor(
    random_state=42
)
```

Gradient Boosting builds an ensemble of sequential decision trees, with later trees attempting to improve upon the errors made by earlier trees.



# 📏 Evaluation Metrics

The models are evaluated using three regression metrics.

### Mean Absolute Error — MAE

```text
MAE = average absolute difference between
actual and predicted yield
```

Lower values indicate better performance.

### Root Mean Squared Error — RMSE

RMSE penalizes larger prediction errors more heavily than MAE.

Lower values indicate better performance.

### R² Score

R² measures how much of the variance in the target is explained by the model.

A higher value generally indicates better predictive performance.



## 🏆 Model Comparison

The notebook creates a comparison DataFrame containing:

| Model             | MAE | RMSE | R² Score |
| -- | --: | : | -: |
| Linear Regression |   — |    — |        — |
| Random Forest     |   — |    — |        — |
| Gradient Boosting |   — |    — |        — |

The actual values should be populated from the final notebook run rather than hard-coded into the README.

The comparison is generated programmatically using:

```python
results = pd.DataFrame({
    'Model': [...],
    'MAE': [...],
    'RMSE': [...],
    'R2 Score': [...]
})
```



# 📈 Prediction Analysis

The notebook visualizes the relationship between:

```text
Actual Yield
```

and:

```text
Predicted Yield
```

using predictions generated by the Random Forest model.

This provides a visual indication of how closely the predictions track the observed yield values.



# 🌟 Feature Importance

Feature importance is analyzed using a standalone Random Forest model.

The categorical variables are converted using:

```python
pd.get_dummies(X)
```

The Random Forest then calculates an importance score for each encoded feature.

The notebook displays the **top 15 most important features**.

This analysis can help identify which agricultural and environmental factors contribute most strongly to the model's predictions.



# 💾 Model Saving

The trained Random Forest pipeline is saved using Joblib:

```python
import joblib

joblib.dump(
    rf_model,
    'crop_yield_model.pkl'
)
```

The resulting file is:

```text
crop_yield_model.pkl
```

Because the saved object is the entire pipeline, it contains both:

```text
Preprocessing
      +
Random Forest model
```

This is useful because inference does not require manually reproducing the preprocessing steps.



# 🔮 Sample Prediction

The notebook demonstrates prediction using a new agricultural sample:

```python
sample = pd.DataFrame({
    'Region': ['North'],
    'Soil_Type': ['Loam'],
    'Crop': ['Wheat'],
    'Rainfall_mm': [500],
    'Temperature_Celsius': [25],
    'Fertilizer_Used': [1],
    'Irrigation_Used': [1],
    'Weather_Condition': ['Sunny'],
    'Days_to_Harvest': [120]
})
```

The trained Random Forest pipeline is then used:

```python
prediction = rf_model.predict(sample)
```

The output represents the predicted:

```text
Yield (tons per hectare)
```



# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Joblib**
* **Jupyter Notebook**



# 📁 Repository Structure

A clean repository structure could look like:

```text
smart-agriculture-crop-yield-prediction/
│
├── README.md
├── smart-agriculture-crop-yield-prediction.ipynb
│
├── models/
│   └── crop_yield_model.pkl
│
├── data/
│   └── README.md
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── requirements.txt
└── .gitignore
```

The current implementation is contained in the Jupyter Notebook. The `src/` structure is recommended if the project is later converted from an experimental notebook into a production-oriented ML application.



# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/<your-username>/smart-agriculture-crop-yield-prediction.git

cd smart-agriculture-crop-yield-prediction
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
```



# ▶️ Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
smart-agriculture-crop-yield-prediction.ipynb
```

and execute the cells sequentially.


If running locally, replace the dataset path with the location of `crop_yield.csv` on your machine.



# 📊 Project Workflow

```text
                    ┌──────────────────┐
                    │   Crop Dataset   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Data Exploration  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Data Preparation │
                    └────────┬─────────┘
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
           Numerical Features    Categorical Features
                  │                     │
                  ▼                     ▼
             Imputation            Imputation
                  │                     │
                  ▼                     ▼
            Standardization        One-Hot Encoding
                  │                     │
                  └──────────┬──────────┘
                             ▼
                     Train/Test Split
                             │
                             ▼
                ┌───────────────────────┐
                │   Regression Models   │
                ├───────────────────────┤
                │ Linear Regression     │
                │ Random Forest         │
                │ Gradient Boosting     │
                └───────────┬───────────┘
                            │
                            ▼
                    Model Evaluation
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
             MAE           RMSE           R²
              │             │             │
              └─────────────┼─────────────┘
                            ▼
                     Model Comparison
                            │
                            ▼
                    Random Forest Model
                            │
                    ┌───────┴────────┐
                    ▼                ▼
             Feature Importance   Model Saving
                                      │
                                      ▼
                              Sample Prediction
```



# 🔬 Key Technical Highlights

* End-to-end supervised learning workflow.
* Regression-based crop-yield prediction.
* Separate preprocessing for numerical and categorical variables.
* Median imputation for numerical missing values.
* Most-frequent imputation for categorical missing values.
* Standardization of numerical features.
* One-hot encoding of categorical variables.
* Comparison of three regression algorithms.
* MAE, RMSE, and R² evaluation.
* Random Forest feature-importance analysis.
* Complete preprocessing + model pipeline serialization.
* Reproducible train/test split.
* Demonstration of inference on unseen agricultural conditions.



# ⚠️ Current Limitations

The current notebook is primarily an exploratory/model-development implementation. Several improvements would be appropriate before production deployment.

### 1. Single Train/Test Split

The project uses a single 80/20 split.

Cross-validation could provide a more robust estimate of generalization performance.

### 2. Limited Hyperparameter Tuning

The Random Forest uses:

```text
n_estimators = 100
```

while Gradient Boosting largely uses default parameters.

Hyperparameter optimization could potentially improve performance.

### 3. Feature Importance Pipeline

The model's main preprocessing pipeline uses `OneHotEncoder`, while the feature-importance analysis separately uses:

```python
pd.get_dummies(X)
```

For a more rigorous production workflow, feature-importance analysis could be tied directly to the fitted preprocessing pipeline so that the feature names correspond exactly to the model's transformed inputs.

### 4. No Independent Validation Dataset

The project uses train/test splitting but does not maintain a separate validation set.

A stronger experimental setup would use:

```text
Training
Validation
Test
```

or cross-validation during model selection.

### 5. Dataset Path Is Kaggle-Specific

The current notebook assumes the dataset is available within the Kaggle environment.

The path should be parameterized for easier local execution.

### 6. No Production API

The model is saved as a `.pkl` file, but there is currently no API or application layer exposing predictions.

A FastAPI service could be added for real-time inference.



# 🔮 Future Improvements

Possible next steps include:

* Implement K-fold cross-validation.
* Perform hyperparameter optimization with GridSearchCV or RandomizedSearchCV.
* Compare additional algorithms such as XGBoost, LightGBM, CatBoost, and Extra Trees.
* Add prediction intervals/uncertainty estimates.
* Perform deeper feature engineering.
* Analyze interactions between rainfall, temperature, soil, and crop type.
* Add SHAP-based model explainability.
* Build a FastAPI prediction endpoint.
* Containerize the model with Docker.
* Build a dashboard for agricultural users.
* Track model versions and experiments.
* Add automated model testing.
* Deploy the prediction service to a cloud platform.



# 📌 Results

The notebook calculates model performance using:

```text
MAE
RMSE
R² Score
```

# 👨‍💻 Project Status

**Status: Machine Learning experimentation / development**

The project currently implements the complete workflow from agricultural data exploration and preprocessing through model comparison, feature-importance analysis, model serialization, and sample prediction.

The next stage would be converting the notebook into a reproducible and deployable machine learning application.



# 📜 License

Add an appropriate open-source license if this repository is intended for public distribution.

For example:

```text
MIT License
```



## 🙏 Acknowledgements

This project was developed as an exploration of machine learning techniques for agricultural yield prediction using environmental, crop, and farming-related variables.

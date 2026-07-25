# Titanic Passenger Survival Prediction

This repository contains a Machine Learning pipeline that uses **Logistic Regression** to predict passenger survival on the Titanic dataset with **80.45% accuracy**.

## 🚀 Final Results & Metrics
The model was evaluated on a test set (20% of the data) containing 179 passengers:
* **Overall Accuracy:** 80.45%
* **Confusion Matrix Breakdown:**
  * **True Negatives (Correctly predicted deceased):** 89
  * **True Positives (Correctly predicted survived):** 55
  * **False Positives (Incorrectly predicted survived):** 16
  * **False Negatives (Incorrectly predicted deceased):** 19

The model shows balanced errors, proving it is not heavily biased toward a single outcome.

---

## 🛠️ Data Engineering & Modeling Approach

### 1. Data Cleaning & Feature Selection
* **Dropping High-Cardinality Strings:** Text columns unique to individuals (`Name`, `Ticket`, `Cabin`) were dropped because text strings cannot be processed natively by scikit-learn models.
* **Target Isolation:** The dataset was separated into features (`X`) and the target classification variable (`y = Survived`).

### 2. Categorical Variable Encoding
* **One-Hot Encoding:** Categorical text fields like `Sex` and `Embarked` were converted into numeric flags (0s and 1s) using `pd.get_dummies(..., drop_first=True)`. 
* Dropping the first dummy column avoids the multi-collinearity trap (the "dummy variable trap") which can destabilize Logistic Regression weights.

### 3. Training & Validation Setup
* **Train-Test Split:** The dataset was partitioned post-encoding into an 80% training set and a 20% validation test set using a locked seed (`random_state=42`) to guarantee code reproducibility.

### 4. Model Training
* A **Logistic Regression** classifier from `sklearn.linear_model` was fitted onto the numeric training arrays. 
* Iterations were increased (`max_iter=1000`) to guarantee that the underlying mathematical solvers successfully converged on an optimal solution.

---

## 📂 Project Structure
```text
├── titanic_survival_model.py  # Final pipeline code
├── README.md                  # Project overview and metrics
└── train.csv                  # Raw dataset (optional)
```

## 💻 How to Run the Pipeline
1. Install dependencies: `pip install pandas scikit-learn`
2. Run the script: `python titanic_survival_model.py`

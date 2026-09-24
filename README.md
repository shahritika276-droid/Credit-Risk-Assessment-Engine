# 💳 Commercial Credit Risk & Loan Default Scoring Engine

An end-to-end, interactive machine learning application designed to assess borrower default risk and automate underwriting decisions in real time. 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## 📌 Business Case & Executive Summary
Evaluating financial creditworthiness accurately is critical for banking and financial institutions to minimize **Non-Performing Loans (NPL)** while optimizing loan approval velocity. 

This engine ingests applicant financial attributes, scales numerical features, and processes them through an optimized **XGBoost Classifier** trained with cost-sensitive class weighting to address significant class imbalance in credit default datasets.

---

## 🛠️ Technical Architecture & Stack
- **Dashboard & UI:** Streamlit
- **Machine Learning Model:** XGBoost (`XGBClassifier`)
- **Data Transformation:** `StandardScaler` (`scikit-learn`)
- **Artifact Serialization:** `joblib`
- **Data Wrangling:** `pandas`, `numpy`

---

## 📊 Borrower Profile Attributes Evaluated
1. **Revolving Line Utilization Rate:** Total credit balance relative to credit limits.
2. **Borrower Age:** Financial maturity indicator.
3. **Debt-to-Income Ratio (DTI):** Monthly debt payments divided by total monthly income.
4. **Monthly Income ($):** Primary liquidity indicator.
5. **Number of Open Credit Lines:** Overall portfolio exposure.
6. **Past Due Delinquencies (30–59 Days):** Historical default risk proxy.

---

## 🚀 How to Run Locally

### Prerequisites
Ensure Python 3.9+ is installed on your local machine.

### Installation & Execution
```bash
# 1. Clone the repository
git clone [https://github.com/shahritika276-droid/Credit-Risk-Assessment-Engine.git](https://github.com/shahritika276-droid/Credit-Risk-Assessment-Engine.git)

# 2. Navigate to project directory
cd Credit-Risk-Assessment-Engine

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit application
streamlit run app.py

# 📊 E-Commerce Customer Churn & LTV Predictive Analytics Model

An end-to-end customer analytics and machine learning pipeline that processes transactional e-commerce data, computes **Recency-Frequency-Monetary (RFM)** behavioral metrics, trains supervised classification models to predict customer churn, and deploys an interactive **Streamlit** dashboard for executive decision-making.

---

## 📌 Business Overview & Problem Statement

Retaining existing customers is significantly more cost-effective than acquiring new ones. In e-commerce, identifying customers at risk of leaving allows businesses to target them with personalized retention campaigns before they churn. 

This project processes over 1 million raw transactional records, engineers granular customer-level behavioral metrics, segments customers into distinct purchasing cohorts, and utilizes supervised machine learning classifiers to predict churn probability with actionable feature importance rankings.

---

## 🛠️ Detailed Technical Features

- **Data Cleaning & Preprocessing:** Processed raw transactional data from the *Online Retail II* dataset using **Pandas**. Filtered out canceled orders, negative quantities, missing customer IDs, and non-retail adjustments to ensure data integrity.
- **RFM Feature Engineering:** Aggregated line-item transactions to the customer level to compute core behavioral metrics:
  - **Recency ($R$):** Days elapsed since the customer's last purchase relative to the dataset max date.
  - **Frequency ($F$):** Total count of unique purchase transactions per customer.
  - **Monetary Value ($M$):** Total gross expenditure across all transactions.
- **Quantile Segmentation & Churn Labeling:** Assigned quantile scores ($1$–$5$) across RFM dimensions to map customers into actionable cohorts (*Champions*, *Loyal Customers*, *At Risk*, *Hibernating*, *About To Sleep*). Defined churn programmatically using an inactivity threshold (>90 days without a purchase).
- **Supervised Churn Classification:** Trained and evaluated **XGBoost** and **Random Forest** algorithms to classify customer churn risk, evaluating metrics across Precision, Recall, and F1-Score.
- **Feature Importance & Driver Analysis:** Extracted Gini importance and feature weights using Scikit-Learn to identify the exact behavioral drivers (e.g., Recency vs. Order Frequency) driving customer drop-off.
- **Interactive Streamlit Dashboard:** Designed an executive web application featuring high-level KPI cards, interactive segment multi-filters, feature importance visual plots, and raw customer inspection tables.

---

## 🚀 Tech Stack & Libraries

- **Programming Language:** Python 3.13
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Machine Learning & Modeling:** Scikit-Learn, XGBoost, Joblib
- **Data Visualization:** Matplotlib, Seaborn
- **Web Application & UI:** Streamlit
- **Dataset Source:** Online Retail II (Excel Data Sheet)

---

## 📁 Repository Structure

- `app.py` — Main Streamlit application containing executive dashboard layout, interactive sidebars, and plotting logic.
- `retail_project.py` — Core machine learning pipeline handling data ingestion, cleaning, RFM aggregation, model training, and artifact export.
- `rfm_data.csv` — Feature-engineered dataset output storing customer-level RFM scores, segment labels, and binary churn targets.
- `xgb_model.pkl` — Serialized pre-trained XGBoost classification model saved for fast loading in the web app.
- `online_retail_II.xlsx` — Raw transactional e-commerce source dataset.
- `README.md` — Project documentation and setup guide.

---

## ⚙️ Installation & Setup Guide

### 1. Clone the Repository
```bash
git clone [https://github.com/Pokuaa30/Customerchurn_ltv.git](https://github.com/Pokuaa30/Customerchurn_ltv.git)
cd Customerchurn_ltv

```
2. Set Up Virtual Environment
Windows:
```Bash
python -m venv .venv
.venv\Scripts\activate
macOS / Linux:

Bash
python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies
Bash
pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn joblib openpyxl
4. Run Pipeline & Launch Dashboard
Train the machine learning pipeline:

Bash
python retail_project.py
Launch the interactive Streamlit dashboard:

Bash
streamlit run app.py
📈 Executive Dashboard Features
The deployed dashboard provides stakeholders with immediate visibility into customer analytics:

Executive Metrics: High-level overview of Total Customers, Churned Count, Overall Churn Rate (50.9%), and Average Monetary Value.

Segment Risk Filtering: Multi-select control allowing users to isolate high-risk groups like Hibernating or About To Sleep.

Feature Importance Visuals: Graphical representation showing which behavioral factors contribute most to model predictions.

Customer Inspection Table: Granular data viewer providing customer-level Recency, Frequency, Monetary, Segment, and Churn status for targeted marketing lists.
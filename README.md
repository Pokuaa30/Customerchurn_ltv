Markdown
# 📊 E-Commerce Customer Churn & LTV Predictive Analytics Model

An end-to-end customer analytics and machine learning pipeline that processes transactional e-commerce data, computes **Recency-Frequency-Monetary (RFM)** behavioral metrics, trains supervised classification models to predict customer churn, and deploys an interactive **Streamlit** dashboard for executive decision-making.

---

## 📌 Project Overview & Architecture

Retaining existing customers is significantly more cost-effective than acquiring new ones. This project addresses customer retention by analyzing historical transactional data, segmenting customers based on purchasing habits, and identifying high-risk churn groups using Machine Learning.

   +-----------------------------------+
   |   Raw Data (online_retail_II.xlsx) |
   +-----------------------------------+
                     │
                     ▼
   +-----------------------------------+
   | Data Cleaning & Feature Eng.      |
   |  (Pandas, RFM Metrics, Segments)  |
   +-----------------------------------+
                     │
                     ▼
   +-----------------------------------+
   | Model Training & Evaluation       |
   |  (Random Forest & XGBoost)        |
   +-----------------------------------+
                     │
                     ▼
   +-----------------------------------+
   | Streamlit Interactive Dashboard   |
   +-----------------------------------+

---

## 🛠️ Key Features

- **Data Cleaning & Engineering:** Cleaned over 1 million transaction records, removed cancellations/refunds, and calculated aggregate customer-level **Recency, Frequency, and Monetary (RFM)** features.
- **Customer Segmentation:** Mapped customers into actionable behavioral segments (e.g., *Champions*, *Loyal Customers*, *At Risk*, *Hibernating*) using quantile scoring.
- **Supervised Churn Modeling:** Trained **XGBoost** and **Random Forest** classifiers to predict customer churn probability.
- **Feature Importance Analysis:** Evaluated feature impact using Scikit-Learn to determine primary churn drivers.
- **Interactive Analytics Dashboard:** Built a responsive **Streamlit** web application for stakeholders to filter high-risk customer segments, review churn rates, and inspect raw customer records.

---

## 🚀 Tech Stack

- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, XGBoost, Joblib
- **Data Visualization:** Matplotlib, Seaborn
- **Dashboard Deployment:** Streamlit

---

## 📁 Repository Structure

cltv_project/
├── .venv/                   # Python Virtual Environment
├── online_retail_II.xlsx    # Raw E-Commerce Transaction Dataset
├── retail_project.py        # Data cleaning, RFM calculation, and model training
├── rfm_data.csv             # Processed dataset with RFM features & churn labels
├── xgb_model.pkl            # Trained XGBoost Machine Learning Model
├── app.py                   # Streamlit Web Application Dashboard
└── README.md                # Project documentation


---

## ⚙️ Installation & Usage

### 1. Clone the Repository
## ⚙️ Installation & Usage

### 1. Clone the Repository
```
bash
git clone https://github.com/YOUR_USERNAME/cltv_project.git
cd cltv_project
2. Create and Activate Virtual Environment
Bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

3. Install Dependencies
Bash
pip install pandas numpy scikit-learn xgboost streamlit matplotlib seaborn joblib openpyxl
4. Run the Data Pipeline & Train Models
Bash
python retail_project.py
5. Launch the Streamlit Dashboard
Bash
streamlit run app.py

📈 Dashboard Preview
The Streamlit interface provides real-time customer analytics:

Executive KPI Metrics: Total Customers, Churned Count, Churn Rate, and Average Monetary Value.

Interactive Risk Filter: Multiselect segment filters to isolate at-risk customer cohorts.

Feature Importance Chart: Model feature scoring indicating key behavioral drivers.

Data Inspection Table: Granular customer-level breakdown for targeted retention campaigns.

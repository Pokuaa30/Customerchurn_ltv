import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# =========================================================
# STEP 1: LOAD & CLEAN DATA
# =========================================================
print("Loading data...")
df_2009_2010 = pd.read_excel('online_retail_II.xlsx', sheet_name='Year 2009-2010')
df_2010_2011 = pd.read_excel('online_retail_II.xlsx', sheet_name='Year 2010-2011')

df = pd.concat([df_2009_2010, df_2010_2011], ignore_index=True)

df_clean = df.dropna(subset=['Customer ID']).copy()
df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['Price'] > 0)]
df_clean['Customer ID'] = df_clean['Customer ID'].astype(int)
df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['Price']

# =========================================================
# STEP 2: BUILD RFM METRICS & SEGMENTS
# =========================================================
snapshot_date = df_clean['InvoiceDate'].max() + dt.timedelta(days=1)

rfm = df_clean.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'Invoice': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5, 4, 3, 2, 1])
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1, 2, 3, 4, 5])
rfm['RF_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str)

segment_map = {
    r'[1-2][1-2]': 'Hibernating',
    r'[1-2][3-4]': 'At_Risk',
    r'[1-2]5': 'Cant_Lose',
    r'3[1-2]': 'About_To_Sleep',
    r'33': 'Need_Attention',
    r'[3-4][4-5]': 'Loyal_Customers',
    r'41': 'Promising',
    r'51': 'New_Customers',
    r'[4-5][2-3]': 'Potential_Loyalists',
    r'5[4-5]': 'Champions'
}
rfm['Segment'] = rfm['RF_Score'].replace(segment_map, regex=True)

# =========================================================
# STEP 3: TRAIN CHURN MACHINE LEARNING MODELS
# =========================================================
# Define Churn: Customer inactive for > 90 days = 1 (Churned), else 0
rfm['Churn'] = (rfm['Recency'] > 90).astype(int)

# Select features & target variable
X = rfm[['Frequency', 'Monetary']]
y = rfm['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

# Train XGBoost
xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])

print("\n--- Random Forest ROC-AUC Score ---", rf_auc)
print("\n--- XGBoost ROC-AUC Score ---", xgb_auc)


# Save RFM dataset and XGBoost model for Streamlit app
rfm.to_csv('rfm_data.csv', index=False)
joblib.dump(xgb_model, 'xgb_model.pkl')
print("\n--- Saved rfm_data.csv and xgb_model.pkl successfully! ---")


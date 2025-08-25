# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix

# --- Step 1: Create a Sample Dataset ---
# In a real-world scenario, you would load your data from a CSV file
# For this example, we'll create a synthetic dataset for demonstration.
np.random.seed(42) # for reproducibility

data = {
    'income': np.random.randint(25000, 150000, 1000),
    'age': np.random.randint(22, 65, 1000),
    'debt': np.random.randint(0, 50000, 1000),
    'months_employed': np.random.randint(1, 120, 1000),
    # Target variable: 1 for 'bad' credit (default), 0 for 'good' credit
    'credit_risk': np.random.choice([0, 1], 1000, p=[0.8, 0.2]) # 80% good, 20% bad
}
df = pd.DataFrame(data)

# Let's make the data more realistic - higher debt and lower income should correlate with higher risk
df.loc[df['debt'] > 40000, 'credit_risk'] = 1
df.loc[df['income'] < 30000, 'credit_risk'] = 1

print("--- First 5 rows of the dataset ---")
print(df.head())
print("\n")


# --- Step 2: Feature Engineering ---
# Create a new feature: debt-to-income ratio
# We add a small number to income to avoid division by zero
df['debt_to_income_ratio'] = df['debt'] / (df['income'] + 1e-6)

print("--- Dataset after feature engineering ---")
print(df.head())
print("\n")


# --- Step 3: Prepare Data for Modeling ---
# Define our features (X) and target (y)
features = ['income', 'age', 'debt', 'months_employed', 'debt_to_income_ratio']
target = 'credit_risk'

X = df[features]
y = df[target]

# Split the data into training (80%) and testing (20%) sets
# This ensures we test our model on data it hasn't seen before
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features. This is important for algorithms like Logistic Regression.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- Step 4: Train and Evaluate Models ---

# Helper function to evaluate and print results
def evaluate_model(model_name, y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    print(f"--- Results for {model_name} ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}  (Correctly predicted defaults / All predicted defaults)")
    print(f"Recall:    {recall:.4f}  (Correctly predicted defaults / All actual defaults)")
    print(f"F1-Score:  {f1:.4f}  (Balance between Precision and Recall)")
    print(f"ROC-AUC:   {roc_auc:.4f}  (Model's ability to distinguish between classes)")
    print("\n")


# --- Model 1: Logistic Regression ---
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Make predictions
lr_preds = lr_model.predict(X_test_scaled)
lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1] # Probabilities for the '1' class

evaluate_model("Logistic Regression", y_test, lr_preds, lr_probs)


# --- Model 2: Random Forest ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# Random Forest is less sensitive to feature scaling, but it's good practice
rf_model.fit(X_train_scaled, y_train) 

# Make predictions
rf_preds = rf_model.predict(X_test_scaled)
rf_probs = rf_model.predict_proba(X_test_scaled)[:, 1]

evaluate_model("Random Forest", y_test, rf_preds, rf_probs)
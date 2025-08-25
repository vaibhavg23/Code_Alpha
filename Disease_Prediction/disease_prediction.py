import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# --- Step 1: Load and Explore the Data ---
# The dataset is hosted by UCI and can be loaded directly.
# This dataset uses a numeric value to represent the presence of heart disease.
# 0 = No disease, 1 = Has disease
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# The dataset does not have a header, so we'll provide column names.
column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

# Load the data into a pandas DataFrame
df = pd.read_csv(url, header=None, names=column_names, na_values='?')

print("--- First 5 rows of the dataset ---")
print(df.head())
print("\n")

# --- Step 2: Prepare the Data ---
# The dataset has some missing values represented by '?'. We'll handle them.
print(f"Number of rows with missing values: {df.isnull().any(axis=1).sum()}")
# For simplicity, we'll drop rows with missing values.
df.dropna(inplace=True)

# Convert the target variable to a binary format (0 or 1)
# Values > 0 indicate some form of heart disease
df['target'] = (df['target'] > 0).astype(int)

# Define our features (X) and the target (y)
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
target = 'target'

X = df[features]
y = df[target]

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Data preprocessing complete.")
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print("\n")


# --- Step 3: Build and Train the Model ---
# We'll use a RandomForestClassifier. This model is an ensemble of decision trees
# and is generally robust and performs well. 🧠
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)
print("Model training completed.")


# --- Step 4: Evaluate the Model ---
# Make predictions on the unseen test data
y_pred = model.predict(X_test)

# Calculate performance metrics 📊
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("--- Model Performance ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}  (Correctly predicted positives / All predicted positives)")
print(f"Recall:    {recall:.4f}  (Correctly predicted positives / All actual positives)")
print(f"F1-Score:  {f1:.4f}  (Balance between Precision and Recall)")
print("\n")
# In medical diagnosis, Recall is often very important. We want to minimize false negatives 
# (i.e., not missing patients who actually have the disease).


# --- Step 5: Feature Importance ---
# Let's see which features the model found most important for prediction.
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

print("--- Top 5 Most Important Features ---")
print(feature_importance_df.head())

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in Heart Disease Prediction")
plt.gca().invert_yaxis()
plt.show()
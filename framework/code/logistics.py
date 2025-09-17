import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

# Step 1. Merge data
df = raw_sample.merge(user_profile, left_on="user", right_on="userid", how="left")

# Step 2. Define label and features
y = df["clk"]   # label: clicked or not
features = ["age_level", "pvalue_level", "shopping_level", "final_gender_code"]
X = df[features].fillna(0)   # handle missing values

# Step 3. Train/Test split (simple random split first)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4. Train a logistic regression model
model = LogisticRegression(max_iter=100)
model.fit(X_train, y_train)

# Step 5. Predictions
y_pred_proba = model.predict_proba(X_test)[:,1]

# Step 6. Evaluation
auc = roc_auc_score(y_test, y_pred_proba)
ll = log_loss(y_test, y_pred_proba)

print(f"AUC: {auc:.4f}")
print(f"LogLoss: {ll:.4f}")

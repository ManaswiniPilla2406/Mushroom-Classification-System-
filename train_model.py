# =============================================
# train_model.py
# This file trains the ML model and saves it
# =============================================

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# ── Step 1: Load Dataset ──────────────────────────────
print("Step 1: Loading dataset...")
df = pd.read_csv("dataset/mushrooms.csv")
print(f"Dataset loaded! Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# ── Step 2: Encode all text columns to numbers ────────
# Machine Learning only understands numbers, not text
# So we convert: 'e' -> 0, 'p' -> 1, 'convex' -> 2, etc.
print("\nStep 2: Encoding text to numbers...")
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)
print("Encoding done!")

# ── Step 3: Split into Features (X) and Target (y) ───
# X = all columns except 'class'
# y = only the 'class' column (edible or poisonous)
print("\nStep 3: Splitting features and target...")
X = df_encoded.drop("class", axis=1)
y = df_encoded["class"]
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# ── Step 4: Split into Train and Test sets ────────────
# 80% for training, 20% for testing
print("\nStep 4: Train-Test Split (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ── Step 5: Train Decision Tree Model ────────────────
print("\nStep 5: Training Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print(f"Decision Tree Accuracy: {dt_acc * 100:.2f}%")

# ── Step 6: Train Random Forest Model ────────────────
print("\nStep 6: Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest Accuracy: {rf_acc * 100:.2f}%")

# ── Step 7: Show Full Report ──────────────────────────
print("\n--- Classification Report (Random Forest) ---")
print(classification_report(y_test, rf_pred, target_names=["Edible", "Poisonous"]))

# ── Step 8: Save the Best Model ──────────────────────
print("Step 8: Saving Random Forest model...")
os.makedirs("models", exist_ok=True)
pickle.dump(rf_model, open("models/model.pkl", "wb"))

# Also save the label encoder (needed for predictions later)
pickle.dump(le, open("models/encoder.pkl", "wb"))

# Save column names (needed for prediction form)
pickle.dump(list(X.columns), open("models/columns.pkl", "wb"))

print("Model saved to models/model.pkl")
print("\nAll done! Now run: python app.py")
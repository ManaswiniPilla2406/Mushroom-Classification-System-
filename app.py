# app.py - Run this file to start the website

from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ── Train model when app starts ───────────────────────
print("Loading and training model...")

df = pd.read_csv("dataset/mushrooms.csv")

# Encode all columns
le_dict = {}
df_encoded = df.copy()
for col in df.columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    le_dict[col] = le

X = df_encoded.drop("class", axis=1)
y = df_encoded["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("Model ready!")

# ── Home page ─────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html", prediction=None, message=None, confidence=None)

# ── Predict ───────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get form values
        input_data = {
            "cap-shape":         request.form["cap_shape"],
            "cap-surface":       "s",
            "cap-color":         request.form["cap_color"],
            "bruises":           request.form["bruises"],
            "odor":              request.form["odor"],
            "gill-attachment":   "f",
            "gill-spacing":      "c",
            "gill-size":         "b",
            "gill-color":        request.form["gill_color"],
            "stalk-shape":       "e",
            "stalk-root":        "e",
            "stalk-surface-above-ring": "s",
            "stalk-surface-below-ring": "s",
            "stalk-color-above-ring":   "w",
            "stalk-color-below-ring":   "w",
            "veil-type":         "p",
            "veil-color":        "w",
            "ring-number":       "o",
            "ring-type":         "p",
            "spore-print-color": request.form["spore_color"],
            "population":        request.form["population"],
            "habitat":           request.form["habitat"],
        }

        # Encode each value using the same encoder
        encoded_row = []
        for col in X.columns:
            val = input_data[col]
            le = le_dict[col]
            if val in le.classes_:
                encoded_val = le.transform([val])[0]
            else:
                encoded_val = 0
            encoded_row.append(encoded_val)

        # Predict
        result     = model.predict([encoded_row])[0]
        proba      = model.predict_proba([encoded_row])[0]
        confidence = round(max(proba) * 100, 1)

        if result == 0:
            prediction = "EDIBLE"
            message    = "This mushroom appears safe to eat."
        else:
            prediction = "POISONOUS"
            message    = "WARNING! This mushroom is likely poisonous."

    except Exception as e:
        prediction = "ERROR"
        message    = f"Something went wrong: {str(e)}"
        confidence = 0

    return render_template(
        "index.html",
        prediction=prediction,
        message=message,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
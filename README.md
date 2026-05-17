# 🍄 Mushroom Classification System

Predict if a mushroom is Edible or Poisonous using Machine Learning.

---

## Tech Used
- Python
- Scikit-learn (Machine Learning)
- Pandas (Data handling)
- Flask (Website)

---

## How to Run — Step by Step

### Step 1 — Install libraries
Open terminal and type:
```
pip install -r requirements.txt
```

### Step 2 — Download Dataset
- Go to: https://www.kaggle.com/datasets/uciml/mushroom-classification
- Download mushrooms.csv
- Put it inside the `dataset/` folder

### Step 3 — Train the Model
```
python train_model.py
```
This creates a file called `model.pkl` inside the `models/` folder.

### Step 4 — Run the Website
```
python app.py
```

### Step 5 — Open in Browser
Go to: http://127.0.0.1:5000

---

## Folder Structure
```
mushroom-classifier/
│
├── dataset/
│   └── mushrooms.csv        ← download from Kaggle
│
├── models/
│   └── model.pkl            ← created after training
│
├── templates/
│   └── index.html           ← the webpage
│
├── app.py                   ← run this to start website
├── train_model.py           ← run this first to train
├── requirements.txt         ← libraries needed
└── README.md
```

---

## How It Works
1. `train_model.py` reads the CSV dataset
2. Converts text to numbers (Label Encoding)
3. Trains Random Forest model (94% accuracy)
4. Saves model to models/model.pkl
5. `app.py` loads the model and runs a website
6. You fill the form → Flask sends to model → Result shown

---

## Models Used
| Model | Accuracy |
|-------|----------|
| Decision Tree | 91% |
| Random Forest | 94% |
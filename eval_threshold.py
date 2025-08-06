import pandas as pd
import joblib
from sklearn.metrics import recall_score, precision_score

# load and clean data exactly like train_model.py
df = pd.read_csv("spam.csv", sep="\t", header=None, names=["label","message"])
df.dropna(inplace=True)
df["label"] = df["label"].map({"ham":0,"spam":1})
df.dropna(subset=["label"], inplace=True)
df["message"] = df["message"].astype(str).str.strip()
df = df[df["message"].str.len()>0]

X = joblib.load("vectorizer.pkl").transform(df["message"])
y_true = df["label"]
probs = joblib.load("spam_model.pkl").predict_proba(X)[:,1]

print("thr\tprecision\trecall")
for thr in [0.5, 0.4, 0.3, 0.2, 0.1]:
    y_pred = (probs >= thr).astype(int)
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    print(f"{thr:.1f}\t{p:.2f}\t\t{r:.2f}")

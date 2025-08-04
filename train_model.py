import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load tab-separated file: first column = label ('ham' or 'spam'), second column = message
df = pd.read_csv("spam.csv", sep="\t", header=None, names=["label", "message"])

# Drop rows with missing label or message
df.dropna(inplace=True)

# Convert label to numeric: ham = 0, spam = 1
df["label"] = df["label"].map({"ham": 0, "spam": 1})

# Drop rows where the label is not 'ham' or 'spam'
df.dropna(subset=["label"], inplace=True)

# Clean message text: convert to string, remove whitespace, drop empty strings
df["message"] = df["message"].astype(str).str.strip()
df = df[df["message"].str.len() > 0]

# Print the final row count after cleanup
print(f"📊 Cleaned dataset size: {len(df)} rows")

# Stop the program if no data is left
if df.empty:
    raise ValueError("🚫 All messages were filtered out. No data left to train on.")

# Convert messages to bag-of-words features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["message"])
y = df["label"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate and print performance
y_pred = model.predict(X_test)
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Save model and vectorizer for future use
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully.")
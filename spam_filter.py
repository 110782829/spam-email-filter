import argparse
import joblib

# Load the saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Define a function that takes a message and returns the prediction
def predict_with_threshold(msg, threshold=0.4):
    # transform text → feature vector
    vec = vectorizer.transform([msg])
    # get probability of “spam” class (index 1)
    spam_prob = model.predict_proba(vec)[0][1]
    label = "Spam" if spam_prob >= threshold else "Not Spam"
    return label, spam_prob

# Set up command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spam filter v2: adjustable threshold"
    )
    parser.add_argument("message", type=str, help="Text to classify")
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.3,
        help="Probability cutoff for marking Spam (default 0.3)"
    )
    args = parser.parse_args()

    label, prob = predict_with_threshold(args.message, args.threshold)
    print(f"Prediction: {label}  (spam probability ≈ {prob:.4f})")
import argparse
import joblib

# Load the saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Define a function that takes a message and returns the prediction
def predict_message(msg):
    # Transform the message into the same feature format used during training
    msg_vector = vectorizer.transform([msg])
    prediction = model.predict(msg_vector)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Set up command-line argument parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a message as Spam or Not Spam")
    parser.add_argument("message", type=str, help="The message you want to classify")
    args = parser.parse_args()

    result = predict_message(args.message)
    print(f"Prediction: {result}")

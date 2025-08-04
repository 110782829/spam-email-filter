# Spam Email Filter

A simple command-line spam message classifier built in Python using scikit-learn.
It uses logistic regression to label messages as Spam or Not Spam.

## Project Structure

```
spam-email-filter/
├── spam.csv             # Dataset (tab-separated)
├── train_model.py       # Trains model and vectorizer
├── spam_filter.py       # CLI tool for message classification
├── spam_model.pkl       # Trained model (saved after training)
├── vectorizer.pkl       # Trained vectorizer (saved after training)
├── .gitignore           # Files to exclude from Git
└── README.md            # Project overview
```

## How to Use

### 1. Install dependencies

Make sure you have Python 3.7+ and install the required libraries:

    pip install pandas scikit-learn joblib

### 2. Train the model

Run the following command:

    python train_model.py

This will:
- Load and clean the dataset
- Train a logistic regression classifier
- Save the model and vectorizer to disk

### 3. Run the spam filter from the command line

Use the trained model to classify a message:

    python spam_filter.py "Congratulations! You've won a free prize!"

Expected output:

    Prediction: Spam

## Dataset

Dataset source:
SMS Spam Collection from the UCI Machine Learning Repository  
https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

The file should be named `spam.csv` and contain two columns: label (`ham` or `spam`) and message.  
Format: **Tab-separated values**

## License

This project is for educational purposes. Dataset provided by the UCI ML Repository.

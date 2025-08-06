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

By default, the script labels as spam if the model probability is ≥ 0.5:

    python spam_filter.py "Congratulations! You've won a free prize!"

Expected output:

    Prediction: Spam  (spam probability ≈ 0.97)

### 4. Evaluate thresholds

To see how spam recall changes at different cutoffs, run:

    python eval_threshold.py

Example output:

    Threshold 0.5: spam recall = 0.91
    Threshold 0.4: spam recall = 0.94
    Threshold 0.3: spam recall = 0.96

#### Custom threshold

You can adjust the spam cutoff threshold with the `-t` flag (e.g. 0.4 to boost recall):

    python spam_filter.py "Free entry in 2 a wkly comp to win tickets!" -t 0.4

## Dataset

Dataset source:  
SMS Spam Collection from the UCI Machine Learning Repository  
https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

The file should be named `spam.csv` and contain two columns: label (`ham` or `spam`) and message.  
Format: **Tab-separated values**

## License

This project is for educational purposes. Dataset provided by the UCI ML Repository.

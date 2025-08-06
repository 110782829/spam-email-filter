# Spam Email Filter (v2.0)

A simple web-based spam message classifier built in Python using scikit-learn and Streamlit.  
It uses logistic regression to label messages as **Spam** or **Not Spam**, with an adjustable probability threshold, and provides both a CLI and a web app interface.

## Project Structure

```
spam-email-filter/
├── spam.csv # Dataset (tab-separated)
├── train_model.py # Trains model and vectorizer
├── spam_filter.py # CLI tool with adjustable threshold
├── eval_threshold.py # Script to benchmark recall vs. threshold
├── app.py # Streamlit web application
├── requirements.txt # Python dependencies
├── spam_model.pkl # Trained model
├── vectorizer.pkl # Trained vectorizer
├── .gitignore # Files to exclude from Git
└── README.md # Project overview
```


## Installation

1. Clone the repository:

    git clone https://github.com/110782829/spam-email-filter.git  
    cd spam-email-filter

2. (Optional) Create and activate a virtual environment:

    python -m venv venv  
    source venv/bin/activate   # macOS/Linux  
    venv\Scripts\activate      # Windows

3. Install dependencies:

    pip install -r requirements.txt

## Training the Model

Run:

    python train_model.py

This will load and clean the dataset, train a logistic regression model, and save the `spam_model.pkl` and `vectorizer.pkl` files.

## Command-Line Interface

Use the CLI to classify a single message:

    python spam_filter.py "Your message here"

By default, a message is labeled **Spam** if its predicted probability ≥ 0.5. To adjust the threshold:

    python spam_filter.py "Your message here" -t 0.4

## Web Application

Launch the Streamlit web app:

    python -m streamlit run app.py

Then open the displayed Local URL (e.g., http://localhost:8501) in your browser. Enter a message, adjust the spam threshold slider, and click **Classify** to see the prediction and probability.

## Evaluating Thresholds

To see how different cutoff values affect spam recall:

    python eval_threshold.py

Example output:

    Threshold 0.5: spam recall = 0.91  
    Threshold 0.4: spam recall = 0.94  
    Threshold 0.3: spam recall = 0.96  

## Dataset

SMS Spam Collection — UCI ML Repository  
https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

Format: Tab-separated values with columns:  
- **label**: `ham` or `spam`  
- **message**: SMS text content  

Save this file as `spam.csv` in the project root.

## License

This project is for educational purposes. Dataset provided by the UCI ML Repository.
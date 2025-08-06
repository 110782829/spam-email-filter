import streamlit as st
import joblib

# load the saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📬 Spam Email Filter")
st.write("Type or paste your message below, adjust the threshold slider, then click **Classify**.")

# user input area
message = st.text_area("Message to classify", height=200)

# threshold slider (0.0 to 1.0, default 0.5)
threshold = st.slider("Spam threshold (probability cutoff)", 0.0, 1.0, 0.5)

if st.button("Classify"):
    # convert message to feature vector and predict
    vec = vectorizer.transform([message])
    spam_prob = model.predict_proba(vec)[0][1]
    label = "🚫 Spam" if spam_prob >= threshold else "✅ Not Spam"

    st.markdown(f"**Prediction:** {label}")
    st.markdown(f"**Spam probability:** {spam_prob:.2f}")

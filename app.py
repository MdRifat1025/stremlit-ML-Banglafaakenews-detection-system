import streamlit as st
import pandas as pd
import joblib
import io

# --- Load model and vectorizer ---
try:
    model = joblib.load("logistic_model.joblib")           # Your saved Logistic Regression
    tfidf = joblib.load("tfidf_vectorizer.joblib")  # Saved TF-IDF vectorizer
    st.sidebar.success("Model and vectorizer loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Failed to load model/vectorizer: {e}")
    model = tfidf = None

# --- Single Text Prediction ---
st.title("Fake News Detection — Logistic Regression")
input_text = st.text_area("Enter article / text to classify:")

if st.button("Predict"):
    if model is None or tfidf is None:
        st.error("Model or vectorizer not loaded.")
    elif input_text.strip() == "":
        st.error("Please enter text to classify.")
    else:
        X_input = tfidf.transform([input_text.strip()])  # transform to 2D
        pred = model.predict(X_input)[0]
        st.write(f"**Prediction:** {pred}")

# --- Batch Prediction ---
st.header("Batch Prediction from CSV")
batch_file = st.file_uploader("Upload CSV (must contain 'text' column)", type=["csv"])

if st.button("Run Batch Prediction"):
    if model is None or tfidf is None:
        st.error("Model or vectorizer not loaded.")
    elif batch_file is None:
        st.error("Upload a CSV file containing a 'text' column.")
    else:
        bdf = pd.read_csv(batch_file)
        if "text" not in bdf.columns:
            st.error("CSV must contain a 'text' column.")
        else:
            X_batch = tfidf.transform(bdf['text'].astype(str))
            preds = model.predict(X_batch)
            bdf['predicted_label'] = preds

            st.write("**Preview of predictions:**")
            st.dataframe(bdf.head())

            # Convert to string for download
            buf = io.StringIO()
            bdf.to_csv(buf, index=False)
            buf.seek(0)
            csv_data = buf.getvalue()

            st.download_button(
                "Download predictions CSV",
                data=csv_data,
                file_name="predictions.csv",
                mime="text/csv"
            )
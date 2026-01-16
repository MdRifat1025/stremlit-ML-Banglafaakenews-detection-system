import streamlit as st
from transformers import AutoTokenizer, TFXLMRobertaForSequenceClassification
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import plotly.express as px  # Fixed typo here

# -------------------- App Config --------------------
st.set_page_config(page_title="Bengali Fake News Detection", layout="wide", page_icon="📰")

# -------------------- Custom CSS --------------------
st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    .stAlert { box-shadow: 2px 2px 5px #ccc; }
    </style>
    """, unsafe_allow_html=True)

st.title("📰 Bengali Fake News Detection System")
st.markdown("---")

# -------------------- Label Map --------------------
label_map = {0: "Real News", 1: "Fake News"}

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    # Ensure this path exists or change to huggingface model ID
    model_dir = "bert_fake_news_model" 
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = TFXLMRobertaForSequenceClassification.from_pretrained(model_dir)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure 'bert_fake_news_model' folder exists.")
        return None, None

with st.spinner("🚀 Loading AI Model... Please wait..."):
    tokenizer, model = load_model()

# -------------------- Sidebar --------------------
st.sidebar.title("⚙️ Settings")
st.sidebar.info("This model uses **BanglaBERT** fine-tuned on Bengali news datasets.")
clear_history = st.sidebar.button("🗑️ Clear History")

if "history" not in st.session_state or clear_history:
    st.session_state.history = []

# -------------------- Main Input --------------------
col1, col2 = st.columns([2, 1])

with col1:
    text = st.text_area("✍️ Enter Bengali News:", height=200, placeholder="এখানে খবরের সত্যতা যাচাই করতে টেক্সট পেস্ট করুন...")
    
    # Text Stats
    if text:
        word_count = len(text.split())
        st.caption(f"Word Count: {word_count}")

with col2:
    st.markdown("### 💡 Instructions")
    st.info("""
    1. Paste the news body in the text box.
    2. Click **Predict**.
    3. Check the confidence score.
    """)

# -------------------- Prediction Logic --------------------
if st.button("🚀 Verify News", type="primary"):
    if not tokenizer or not model:
        st.error("Model not loaded correctly.")
    elif text.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        # Progress bar animation
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            progress_bar.progress(percent_complete + 1)
        
        # Tokenize (Using padding=True is often faster for single inference than max_length)
        inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True, max_length=512)

        # Prediction
        outputs = model(inputs)
        logits = outputs.logits
        pred = int(tf.argmax(logits, axis=1).numpy()[0])
        prob = tf.nn.softmax(logits, axis=1).numpy()[0]
        confidence = float(np.max(prob))
        
        # Determine Color
        pred_label = label_map[pred]
        color_code = "#28a745" if pred == 0 else "#dc3545" # Green for Real, Red for Fake

        # -------------------- Display Results --------------------
        st.markdown("---")
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            st.markdown(f"<h2 style='text-align: center; color: {color_code}; border: 2px solid {color_code}; padding: 10px; border-radius: 10px;'>{pred_label}</h2>", unsafe_allow_html=True)
            st.metric(label="Confidence Level", value=f"{confidence*100:.2f}%", delta="High Confidence" if confidence > 0.8 else "Low Confidence")

            if confidence < 0.6:
                st.warning("⚠️ Model is uncertain. Cross-verify with other sources.")

        with res_col2:
            # Interactive Chart
            chart_data = pd.DataFrame({
                "Category": ["Real News", "Fake News"],
                "Probability": prob
            })
            fig = px.bar(chart_data, x='Category', y='Probability', color='Category', 
                         color_discrete_map={"Real News": "green", "Fake News": "red"}, height=300)
            st.plotly_chart(fig, use_container_width=True)

        # -------------------- Update History --------------------
        st.session_state.history.insert(0, {
            "text": text[:50] + "...", 
            "predicশেখ হাসিনা বাংলাদেশে ফেরত আসছে tion": pred_label, 
            "confidence": f"{confidence*100:.2f}%",
            "time": pd.Timestamp.now().strftime("%H:%M:%S")
        })

# -------------------- History & Downloads --------------------
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Recent Checks")
    
    # Display as a styled dataframe
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)

    d_col1, d_col2 = st.columns(2)
    with d_col1:
        st.download_button(
            "⬇️ Download History (JSON)",
            json.dumps(st.session_state.history, ensure_ascii=False),
            file_name="news_history.json",
            mime="application/json"
        )

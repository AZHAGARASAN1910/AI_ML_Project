import streamlit as st
import pandas as pd
import re
import pickle
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)


# Load NLTK
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Load models
@st.cache_resource
def load_models():

    logistic_model = joblib.load("logistic_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")

    # Load RNN model
    rnn_model = load_model(
        "fake_news_rnn.keras",
        compile=False
    )

    # Load LSTM model
    lstm_model = load_model(
        "fake_news_lstm.keras",
        compile=False
    )

    # Load tokenizer
    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    return (
        logistic_model,
        tfidf,
        rnn_model,
        lstm_model,
        tokenizer
    )


logistic_model, tfidf, rnn_model, lstm_model, tokenizer = load_models()


# Text preprocessing
def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"<.*?>", " ", text)

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    tokens = text.split()

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# Title
st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article to classify it using "
    "Logistic Regression, RNN and LSTM."
)

st.divider()


# Input
title = st.text_input(
    "News Title",
    placeholder="Enter news title"
)

text = st.text_area(
    "News Content",
    height=250,
    placeholder="Paste news content here..."
)


# Detect button
if st.button(
    "🔍 Detect Fake News",
    type="primary",
    use_container_width=True
):

    if title.strip() == "" and text.strip() == "":

        st.warning(
            "Please enter a news title or news content."
        )

    else:

        news = title + " " + text

        processed_text = clean_text(news)

        # -------------------------------
        # Logistic Regression
        # -------------------------------

        tfidf_input = tfidf.transform(
            [processed_text]
        )

        logistic_probability = (
            logistic_model.predict_proba(
                tfidf_input
            )[0][1]
        )

        logistic_result = (
            "REAL"
            if logistic_probability >= 0.5
            else "FAKE"
        )


        # -------------------------------
        # RNN
        # -------------------------------

        rnn_sequence = tokenizer.texts_to_sequences(
            [processed_text]
        )

        rnn_padded = pad_sequences(
            rnn_sequence,
            maxlen=300,
            padding="post",
            truncating="post"
        )

        rnn_probability = (
            rnn_model.predict(
                rnn_padded,
                verbose=0
            )[0][0]
        )

        rnn_result = (
            "REAL"
            if rnn_probability >= 0.5
            else "FAKE"
        )


        # -------------------------------
        # LSTM
        # -------------------------------

        lstm_sequence = tokenizer.texts_to_sequences(
            [processed_text]
        )

        lstm_padded = pad_sequences(
            lstm_sequence,
            maxlen=300,
            padding="post",
            truncating="post"
        )

        lstm_probability = (
            lstm_model.predict(
                lstm_padded,
                verbose=0
            )[0][0]
        )

        lstm_result = (
            "REAL"
            if lstm_probability >= 0.5
            else "FAKE"
        )


        # -------------------------------
        # Results
        # -------------------------------

        st.divider()

        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)


        # Logistic
        with col1:

            st.markdown("### 📊 Logistic Regression")

            if logistic_result == "REAL":
                st.success("✅ REAL NEWS")
            else:
                st.error("❌ FAKE NEWS")

            st.write(
                f"Probability: {logistic_probability:.2%}"
            )

            st.progress(
                float(logistic_probability)
            )


        # RNN
        with col2:

            st.markdown("### 🧠 RNN")

            if rnn_result == "REAL":
                st.success("✅ REAL NEWS")
            else:
                st.error("❌ FAKE NEWS")

            st.write(
                f"Probability: {rnn_probability:.2%}"
            )

            st.progress(
                float(rnn_probability)
            )


        # LSTM
        with col3:

            st.markdown("### 🔥 LSTM")

            if lstm_result == "REAL":
                st.success("✅ REAL NEWS")
            else:
                st.error("❌ FAKE NEWS")

            st.write(
                f"Probability: {lstm_probability:.2%}"
            )

            st.progress(
                float(lstm_probability)
            )


        # -------------------------------
        # Comparison table
        # -------------------------------

        st.divider()

        st.subheader("Model Comparison")

        results = pd.DataFrame({
            "Model": [
                "Logistic Regression",
                "RNN",
                "LSTM"
            ],

            "Prediction": [
                logistic_result,
                rnn_result,
                lstm_result
            ],

            "Probability": [
                f"{logistic_probability:.2%}",
                f"{rnn_probability:.2%}",
                f"{lstm_probability:.2%}"
            ]
        })

        st.dataframe(
            results,
            use_container_width=True,
            hide_index=True
        )


        # -------------------------------
        # Final majority result
        # -------------------------------

        predictions = [
            logistic_result,
            rnn_result,
            lstm_result
        ]

        real_count = predictions.count("REAL")
        fake_count = predictions.count("FAKE")

        st.divider()

        st.subheader("Final Result")

        if real_count > fake_count:

            st.success(
                f"✅ REAL NEWS "
                f"({real_count}/3 models)"
            )

        elif fake_count > real_count:

            st.error(
                f"❌ FAKE NEWS "
                f"({fake_count}/3 models)"
            )

        else:

            st.warning(
                "⚠️ Models produced different predictions."
            )
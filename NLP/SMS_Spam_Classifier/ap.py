import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# -----------------------------
# NLTK Downloads
# -----------------------------
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")


# -----------------------------
# Initialize Stemmer
# -----------------------------
ps = PorterStemmer()


# -----------------------------
# Text Transformation Function
# -----------------------------
def transform_text(text):

    # Convert to lowercase
    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    # Keep only alphanumeric words
    y = []

    for word in text:
        if word.isalnum():
            y.append(word)

    # Remove stopwords and punctuation
    text = y[:]
    y.clear()

    for word in text:
        if word not in stopwords.words("english") and word not in string.punctuation:
            y.append(word)

    # Stemming
    text = y[:]
    y.clear()

    for word in text:
        y.append(ps.stem(word))

    # Convert list to string
    return " ".join(y)


# -----------------------------
# Load Vectorizer and Model
# -----------------------------

vectorizer_path = r"C:\Users\Ezhilarasi\OneDrive\Desktop\NLP PROJECT\New folder\vectorizer.pkl"

model_path = r"C:\Users\Ezhilarasi\OneDrive\Desktop\NLP PROJECT\New folder\model.pkl"


with open(vectorizer_path, "rb") as file:
    tfidf = pickle.load(file)


with open(model_path, "rb") as file:
    model = pickle.load(file)


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧"
)


st.title("📧 Email/SMS Spam Classifier")

st.write("Enter an SMS or email message below and click Predict.")


# Text input
input_sms = st.text_area(
    "Enter the message:",
    height=150,
    placeholder="Example: Congratulations! You have won a free prize..."
)


# Predict button
if st.button("Predict"):

    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message.")

    else:

        # -----------------------------
        # 1. Preprocess
        # -----------------------------
        transformed_sms = transform_text(input_sms)

        # -----------------------------
        # 2. Vectorize
        # -----------------------------
        vector_input = tfidf.transform([transformed_sms])

        # -----------------------------
        # 3. Prediction
        # -----------------------------
        result = model.predict(vector_input)[0]

        # -----------------------------
        # 4. Display Result
        # -----------------------------

        if result == 1:
            st.error("🚨 SPAM")

        else:
            st.success("✅ NOT SPAM")
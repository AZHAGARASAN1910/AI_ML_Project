import json, pickle, re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ART = Path(__file__).resolve().parent / "artifacts"
st.set_page_config(page_title="Customer Churn & Sentiment", page_icon="📊", layout="wide")

# Fallback numbers (from your notebook run) if ml_metrics.json is missing
FALLBACK_ML = {
    "Logistic Regression": dict(accuracy=77.15, precision=58.90, recall=45.99, f1=51.65, roc_auc=0.8149),
    "KNN":                 dict(accuracy=74.52, precision=52.20, recall=47.59, f1=49.79, roc_auc=0.7586),
    "Random Forest":       dict(accuracy=75.80, precision=54.65, recall=51.87, f1=53.22, roc_auc=0.7787),
    "SVM":                 dict(accuracy=77.64, precision=59.61, recall=48.93, f1=53.74, roc_auc=0.7733),
}

@st.cache_resource
def load_ml():
    return (pickle.load(open(ART / "tfidf.pkl", "rb")),
            pickle.load(open(ART / "ml_models.pkl", "rb")),
            json.load(open(ART / "categorical_options.json")))

@st.cache_resource
def load_dl():
    from tensorflow.keras.models import load_model
    return (load_model(ART / "rnn_model.keras"), load_model(ART / "lstm_model.keras"),
            pickle.load(open(ART / "tokenizer.pkl", "rb")),
            pickle.load(open(ART / "label_encoder.pkl", "rb")))

def load_json(name, default=None):
    p = ART / name
    return json.load(open(p)) if p.exists() else default

def clean_text(t):
    t = t.lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"[^a-z0-9\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()

st.title("📊 Customer Churn & Sentiment Analysis")
tab1, tab2, tab3 = st.tabs(["Churn prediction (ML)", "Feedback sentiment (RNN / LSTM)", "Model performance"])

# ---------------- Tab 1: ML churn ----------------
with tab1:
    st.subheader("Predict customer churn")
    try:
        tfidf, models, options = load_ml()
    except Exception as e:
        st.error(f"ML artifacts not found in ./artifacts ({e}). Run save_ml() in your notebook first.")
    else:
        model_name = st.selectbox("Model", list(models.keys()), index=3)
        cols = st.columns(3)
        values = {}
        for i, (col, opts) in enumerate(options.items()):
            values[col] = cols[i % 3].selectbox(col, opts, key=f"ml_{col}")
        if st.button("Predict churn", type="primary"):
            # same text construction as the notebook: join categorical values with spaces
            text = " ".join(str(values[c]) for c in options)
            model = models[model_name]
            pred = int(model.predict(tfidf.transform([text]))[0])
            if pred == 1:
                st.error("⚠️ Likely to CHURN")
            else:
                st.success("✅ Likely to STAY")
            if hasattr(model, "predict_proba"):
                st.metric("Churn probability", f"{model.predict_proba(tfidf.transform([text]))[0][1]*100:.1f}%")
        st.caption("Note: these models use only the categorical columns (as TF-IDF text), exactly as in the notebook.")

# ---------------- Tab 2: DL sentiment ----------------
with tab2:
    st.subheader("Classify customer feedback")
    try:
        rnn, lstm, tok, le = load_dl()
    except Exception as e:
        st.error(f"DL artifacts not found in ./artifacts ({e}). Run save_dl() in your notebook first.")
    else:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        maxlen = (load_json("dl_metrics.json") or {}).get("max_length", 100)
        choice = st.radio("Model", ["LSTM", "RNN", "Both"], horizontal=True)
        text = st.text_area("Customer feedback", "The service was slow and support never helped me.", height=120)
        if st.button("Analyze sentiment", type="primary") and text.strip():
            seq = pad_sequences(tok.texts_to_sequences([clean_text(text)]),
                                maxlen=maxlen, padding="post", truncating="post")
            for name, m in [("LSTM", lstm), ("RNN", rnn)]:
                if choice in (name, "Both"):
                    prob = m.predict(seq, verbose=0)[0]
                    label = le.classes_[int(np.argmax(prob))]
                    st.markdown(f"**{name}:** {label} ({prob.max()*100:.1f}%)")
                    st.bar_chart(pd.DataFrame({"probability": prob}, index=[str(c) for c in le.classes_]))

# ---------------- Tab 3: metrics ----------------
with tab3:
    st.subheader("Machine learning models (churn)")
    ml = load_json("ml_metrics.json", FALLBACK_ML)
    for k in ml:  # add ROC-AUC from fallback if the saved file does not have it
        ml[k].setdefault("roc_auc", FALLBACK_ML.get(k, {}).get("roc_auc"))
    df = pd.DataFrame({k: {m: v[m] for m in ("accuracy", "precision", "recall", "f1")} for k, v in ml.items()}).T
    df.columns = ["Accuracy %", "Precision %", "Recall %", "F1 %"]
    df["ROC-AUC"] = [ml[k]["roc_auc"] for k in ml]
    st.dataframe(df.round(4), use_container_width=True)
    st.bar_chart(df.drop(columns="ROC-AUC"))
    best = df["ROC-AUC"].idxmax()
    st.caption(f"Best ROC-AUC: {best} ({df.loc[best, 'ROC-AUC']:.4f}). Best accuracy: {df['Accuracy %'].idxmax()}.")
    conf = {k: v["confusion"] for k, v in ml.items() if "confusion" in v}
    if conf:
        pick = st.selectbox("Confusion matrix", list(conf))
        st.dataframe(pd.DataFrame(conf[pick], index=["Actual No Churn", "Actual Churn"],
                                  columns=["Pred No Churn", "Pred Churn"]))

    st.subheader("Deep learning models (sentiment)")
    dl = load_json("dl_metrics.json")
    if dl:
        st.dataframe(pd.DataFrame({"Accuracy %": {"RNN": dl["RNN"]["accuracy"], "LSTM": dl["LSTM"]["accuracy"]}}).round(2))
        pick = st.selectbox("Classification report", ["LSTM", "RNN"])
        rep = pd.DataFrame(dl[pick]["report"]).T
        st.dataframe(rep.round(3), use_container_width=True)
    else:
        st.info("Run save_dl() in the notebook to show RNN/LSTM results.")

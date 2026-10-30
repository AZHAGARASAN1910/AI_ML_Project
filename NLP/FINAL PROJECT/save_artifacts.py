# ---- PASTE THIS AS A NEW CELL AT THE END OF YOUR NOTEBOOK AND RUN IT ----
# It saves everything the Streamlit app needs into an "artifacts" folder.
import os, json, pickle
os.makedirs("artifacts", exist_ok=True)

# ---------- ML part (Telco churn) ----------
# NOTE: variables below come from your ML cells. The notebook reuses the names
# X, y, X_train, ... for the DL part, so re-run the ML cells (1-21) first if needed,
# OR run this ML block right after cell 21 and the DL block after cell 44.
def save_ml(df, categorical_cols, tfidf, models, X_test_ml, y_test_ml):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, confusion_matrix)
    metrics = {}
    for name, m in models.items():
        p = m.predict(X_test_ml)
        metrics[name] = dict(
            accuracy=accuracy_score(y_test_ml, p) * 100,
            precision=precision_score(y_test_ml, p, zero_division=0) * 100,
            recall=recall_score(y_test_ml, p, zero_division=0) * 100,
            f1=f1_score(y_test_ml, p, zero_division=0) * 100,
            confusion=confusion_matrix(y_test_ml, p).tolist())
    pickle.dump(tfidf, open("artifacts/tfidf.pkl", "wb"))
    pickle.dump(models, open("artifacts/ml_models.pkl", "wb"))
    meta = {c: sorted(df[c].dropna().astype(str).unique().tolist()) for c in categorical_cols}
    json.dump(meta, open("artifacts/categorical_options.json", "w"))
    json.dump(metrics, open("artifacts/ml_metrics.json", "w"))

# Run right after cell 21 (ML section):
# save_ml(df, categorical_cols, tfidf, models, X_test, y_test)

# ---------- DL part (sentiment RNN / LSTM) ----------
def save_dl(tokenizer, le, rnn_model, lstm_model, y_test_dl, rnn_pred, lstm_pred, max_length=100):
    from sklearn.metrics import accuracy_score, classification_report
    rnn_model.save("artifacts/rnn_model.keras")
    lstm_model.save("artifacts/lstm_model.keras")
    pickle.dump(tokenizer, open("artifacts/tokenizer.pkl", "wb"))
    pickle.dump(le, open("artifacts/label_encoder.pkl", "wb"))
    out = {"max_length": max_length}
    for n, p in [("RNN", rnn_pred), ("LSTM", lstm_pred)]:
        out[n] = dict(accuracy=accuracy_score(y_test_dl, p) * 100,
                      report=classification_report(y_test_dl, p, output_dict=True,
                                                   target_names=[str(c) for c in le.classes_]))
    json.dump(out, open("artifacts/dl_metrics.json", "w"))

# Run right after cell 44 (DL section):
# save_dl(tokenizer, le, rnn_model, lstm_model, y_test, rnn_pred, lstm_pred)

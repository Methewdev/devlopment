import streamlit as st
import torch
import re
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Analisis Emosi & Sarkasme", layout="centered")
st.title("📊 Analisis Sentimen, Emosi & Sarkasme")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    model_name = "envidevelopment/sentiment-banking"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# =========================
# PREPROCESS
# =========================
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =========================
# PREDICT
# =========================
def predict(text):
    clean_text = preprocess(text)
    inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    confidence = torch.max(probs).item()

    label = model.config.id2label[pred]
    return label, confidence

# =========================
# DETEKSI SARKASME
# =========================
def detect_sarcasm(text, emotion):
    text = text.lower()

    if "menguji kesabaran" in text:
        return "Ya"

    if "luar biasa" in text and "gagal" in text:
        return "Ya"

    if emotion in ["senang", "netral"] and any(k in text for k in ["error", "gagal", "lambat"]):
        return "Ya"

    return "Tidak"

# =========================
# 🔥 LOAD CSV (FIX TOTAL)
# =========================
def load_csv(uploaded_file):
    for enc in ["utf-8", "latin-1", "cp1252"]:
        for sep in [",", ";", "\t"]:
            try:
                uploaded_file.seek(0)  # 🔥 WAJIB
                df = pd.read_csv(uploaded_file, encoding=enc, sep=sep)
                return df
            except:
                continue
    return None

# =========================
# SINGLE INPUT
# =========================
st.markdown("## ✍️ Analisis Satu Kalimat")
text_input = st.text_area("Masukkan teks")

if st.button("🔍 Analisis"):
    if text_input:
        emotion, conf = predict(text_input)
        sarcasm = detect_sarcasm(text_input, emotion)

        st.write("### Hasil:")
        st.write("Emosi:", emotion)
        st.write("Confidence:", round(conf, 2))
        st.write("Sarkasme:", sarcasm)

# =========================
# BULK UPLOAD
# =========================
st.markdown("---")
st.markdown("## 📂 Analisis Bulk CSV")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    df = load_csv(uploaded_file)

    if df is None:
        st.error("❌ File tidak bisa dibaca. Pastikan CSV valid.")
        st.stop()

    st.write("Preview data:")
    st.dataframe(df.head())

    # auto detect kolom teks
    text_col = None
    for col in ["content", "text", "ulasan", "review"]:
        if col in df.columns:
            text_col = col
            break

    if text_col is None:
        st.error("❌ Kolom teks tidak ditemukan")
    else:
        if st.button("🚀 Proses"):
            emotions, sarcasms, confidences = [], [], []

            for text in df[text_col]:
                emotion, conf = predict(text)
                sarcasm = detect_sarcasm(text, emotion)

                emotions.append(emotion)
                sarcasms.append(sarcasm)
                confidences.append(conf)

            df["emotion"] = emotions
            df["sarcasm"] = sarcasms
            df["confidence"] = confidences

            st.success("✅ Selesai")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download", csv, "hasil.csv")

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
# STYLE EMOSI
# =========================
def get_emotion_style(label):
    styles = {
        "senang": ("😊 Senang", "#4CAF50"),
        "marah": ("😡 Marah", "#F44336"),
        "sedih": ("😢 Sedih", "#2196F3"),
        "kecewa": ("😞 Kecewa", "#FF9800"),
        "netral": ("😐 Netral", "#9E9E9E")
    }
    return styles.get(label, ("❓ Tidak diketahui", "#000000"))

# =========================
# PREDICT EMOSI
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
# DETEKSI SARKASME (HYBRID)
# =========================
def detect_sarcasm(text, emotion):
    text = text.lower()

    positive_words = ["bagus", "mantap", "keren", "hebat", "luar biasa"]
    negative_words = ["error", "gagal", "lambat", "lemot", "tidak bisa", "login gagal"]

    implicit_patterns = [
        "menguji kesabaran",
        "terima kasih ya",
        "hebat ya"
    ]

    score = 0

    # implicit sarcasm
    for p in implicit_patterns:
        if p in text:
            score += 3

    # positive + negative
    for pos in positive_words:
        for neg in negative_words:
            if pos in text and neg in text:
                score += 2

    # hybrid emotion
    if emotion in ["senang", "netral"]:
        if any(n in text for n in negative_words):
            score += 2

    return "Ya" if score >= 2 else "Tidak"

# =========================
# LOAD CSV (FIX ENCODING)
# =========================
def load_csv(uploaded_file):
    for enc in ["utf-8", "latin-1", "cp1252"]:
        try:
            return pd.read_csv(uploaded_file, encoding=enc)
        except:
            continue
    return None

# =========================
# SINGLE INPUT
# =========================
st.markdown("## ✍️ Analisis Satu Kalimat")
user_input = st.text_area("Masukkan komentar")

if st.button("🔍 Analisis"):
    if user_input.strip():
        hasil, confidence = predict(user_input)
        label_text, color = get_emotion_style(hasil)
        sarcasm = detect_sarcasm(user_input, hasil)

        st.markdown("### 📌 Hasil Emosi")
        st.markdown(
            f"""
            <div style="background:{color};padding:20px;border-radius:10px;text-align:center;color:white;font-size:24px">
                {label_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(f"Confidence Score: {confidence:.2f}")

        st.markdown("### 🎭 Deteksi Sarkasme")
        if sarcasm == "Ya":
            st.error("😏 Sarkasme Terdeteksi")
        else:
            st.success("🙂 Tidak Sarkasme")

    else:
        st.warning("⚠️ Masukkan teks terlebih dahulu")

# =========================
# BULK UPLOAD
# =========================
st.markdown("---")
st.markdown("## 📂 Analisis Bulk (CSV)")

uploaded_file = st.file_uploader("Upload CSV (harus ada kolom 'content')", type=["csv"])

if uploaded_file:
    df = load_csv(uploaded_file)

    if df is None:
        st.error("❌ File tidak bisa dibaca (encoding error)")
        st.stop()

    if "content" not in df.columns:
        st.error("❌ Kolom 'content' tidak ditemukan")
    else:
        if st.button("🚀 Proses Bulk"):
            emotions = []
            sarcasms = []
            confidences = []

            for text in df["content"]:
                emotion, conf = predict(text)
                sarcasm = detect_sarcasm(text, emotion)

                emotions.append(emotion)
                sarcasms.append(sarcasm)
                confidences.append(conf)

            df["emotion"] = emotions
            df["sarcasm"] = sarcasms
            df["confidence"] = confidences

            st.success("✅ Selesai diproses")
            st.dataframe(df)

            # download
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Hasil", csv, "hasil_analisis.csv")

            # grafik
            st.markdown("## 📊 Distribusi Emosi")
            st.bar_chart(df["emotion"].value_counts())

            st.markdown("## 🎭 Distribusi Sarkasme")
            st.bar_chart(df["sarcasm"].value_counts())

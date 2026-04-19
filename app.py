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
# STYLE EMOSI (UI)
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
# DETEKSI SARKASME (SIMPLE HYBRID)
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
# LOAD CSV (ROBUST)
# =========================
def load_csv(uploaded_file):
    for enc in ["utf-8", "latin-1", "cp1252"]:
        for sep in [",", ";", "\t"]:
            try:
                uploaded_file.seek(0)
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

        label_text, color = get_emotion_style(emotion)

        # =========================
        # OUTPUT EMOSI
        # =========================
        st.markdown("### 📌 Hasil Emosi")

        st.markdown(
            f"""
            <div style="
                background-color:{color};
                padding:20px;
                border-radius:12px;
                text-align:center;
                color:white;
                font-size:26px;
                font-weight:bold;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
            ">
                {label_text}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(f"Confidence: {conf:.2f}")

        # =========================
        # OUTPUT SARKASME
        # =========================
        st.markdown("### 🎭 Sarkasme")

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
st.markdown("## 📂 Analisis Bulk CSV")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    df = load_csv(uploaded_file)

    if df is None:
        st.error("❌ File tidak bisa dibaca")
        st.stop()

    st.write("Preview data:")
    st.dataframe(df.head())

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

            # tambah emoji
            df["emotion_label"] = df["emotion"].map({
                "senang": "😊 Senang",
                "marah": "😡 Marah",
                "sedih": "😢 Sedih",
                "kecewa": "😞 Kecewa",
                "netral": "😐 Netral"
            })

            st.success("✅ Selesai")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download", csv, "hasil.csv")

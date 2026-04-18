import streamlit as st
import torch
import re
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
# PREPROCESS (SAMA DENGAN TRAINING)
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
# DETEKSI SARKASME (HYBRID + IMPLICIT 🔥)
# =========================
def detect_sarcasm(text, emotion):
    text = text.lower()

    positive_words = [
        "bagus", "mantap", "keren", "hebat", "luar biasa", "baik"
    ]

    negative_words = [
        "error", "gagal", "lambat", "lemot", "buruk", "jelek",
        "tidak bisa", "login gagal"
    ]

    # 🔥 implicit sarcasm (penting)
    implicit_patterns = [
        "menguji kesabaran",
        "terima kasih ya",
        "luar biasa sekali",
        "hebat ya"
    ]

    score = 0

    # RULE 1: implicit sarcasm
    for p in implicit_patterns:
        if p in text:
            score += 3

    # RULE 2: positive + negative
    for pos in positive_words:
        for neg in negative_words:
            if pos in text and neg in text:
                score += 2

    # RULE 3: emotion contradiction (HYBRID)
    if emotion in ["senang", "netral"]:
        if any(n in text for n in negative_words):
            score += 2

    if score >= 2:
        return "😏 Sarkasme Terdeteksi"
    
    return "🙂 Tidak Sarkasme"

# =========================
# INPUT USER
# =========================
st.markdown("### ✍️ Masukkan komentar")
user_input = st.text_area("Contoh: Pelayanan mantap banget, tapi sering error")

# =========================
# BUTTON
# =========================
if st.button("🔍 Analisis"):
    if user_input.strip() != "":
        
        # Prediksi emosi
        hasil, confidence = predict(user_input)
        label_text, color = get_emotion_style(hasil)

        # Deteksi sarkasme (FIXED)
        sarcasm_result = detect_sarcasm(user_input, hasil)

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

        st.markdown(f"**Confidence Score:** {confidence:.2f}")

        # =========================
        # OUTPUT SARKASME
        # =========================
        st.markdown("### 🎭 Deteksi Sarkasme")

        if "Terdeteksi" in sarcasm_result:
            st.error(sarcasm_result)
        else:
            st.success(sarcasm_result)

    else:
        st.warning("⚠️ Masukkan teks terlebih dahulu")

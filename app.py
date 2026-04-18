import streamlit as st
import torch
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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)

    pred = torch.argmax(probs).item()
    confidence = torch.max(probs).item()

    label_map = {
        0: "senang",
        1: "marah",
        2: "sedih",
        3: "kecewa",
        4: "netral"
    }

    return label_map[pred], confidence

# =========================
# DETEKSI SARKASME (RULE-BASED)
# =========================
def detect_sarcasm(text):
    text_lower = text.lower()

    sarcasm_keywords = [
        "ya bagus banget",
        "hebat sekali",
        "mantap banget",
        "terima kasih ya",
        "luar biasa sekali",
        "keren banget",
        "top banget"
    ]

    negative_context = [
        "error",
        "lama",
        "buruk",
        "jelek",
        "gagal",
        "lemot",
        "tidak bisa"
    ]

    for s in sarcasm_keywords:
        for n in negative_context:
            if s in text_lower and n in text_lower:
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

        # Deteksi sarkasme
        sarcasm_result = detect_sarcasm(user_input)

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

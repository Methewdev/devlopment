import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# TITLE
# =========================
st.set_page_config(page_title="Analisis Emosi", layout="centered")
st.title("📊 Analisis Sentimen & Emosi")

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
# PREDICT FUNCTION
# =========================
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)

    pred = torch.argmax(probs).item()
    confidence = torch.max(probs).item()

    # Mapping sesuai training kamu
    label_map = {
        0: "senang",
        1: "marah",
        2: "sedih",
        3: "kecewa",
        4: "netral"
    }

    return label_map[pred], confidence

# =========================
# INPUT USER
# =========================
st.markdown("### ✍️ Masukkan komentar")
user_input = st.text_area("Contoh: Pelayanan sangat ramah dan cepat")

# =========================
# BUTTON
# =========================
if st.button("🔍 Prediksi"):
    if user_input.strip() != "":
        hasil, confidence = predict(user_input)

        label_text, color = get_emotion_style(hasil)

        st.markdown("### 📌 Hasil Analisis")

        # BOX HASIL
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

        # CONFIDENCE
        st.markdown(f"**Confidence Score:** {confidence:.2f}")

    else:
        st.warning("⚠️ Masukkan teks terlebih dahulu")

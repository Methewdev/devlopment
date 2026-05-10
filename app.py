import streamlit as st
import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from io import StringIO

# ==========================================
# CONFIG STREAMLIT
# ==========================================
st.set_page_config(
    page_title="Analisis Emosi & Sarkasme",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Analisis Emosi & Sarkasme")
st.markdown("Deteksi Emosi, Sentimen dan Sarkasme pada Ulasan Mobile Banking")

# ==========================================
# LOAD MODEL
# ==========================================
MODEL_NAME = "envidevelopment/sentiment-banking"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_model()

# ==========================================
# LABEL
# ==========================================
id2label = {
    0: "senang",
    1: "marah",
    2: "sedih",
    3: "kecewa",
    4: "netral"
}

# ==========================================
# PREPROCESSING
# ==========================================
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# DETEKSI SARKASME
# ==========================================
def detect_sarcasm(text):

    text = str(text).lower()

    positive_words = [
        "bagus", "mantap", "keren",
        "hebat", "luar biasa",
        "baik", "top", "recommended"
    ]

    negative_words = [
        "error", "gagal", "lemot",
        "buruk", "jelek", "crash",
        "force close", "lambat",
        "tidak bisa"
    ]

    irony_markers = [
        "terima kasih",
        "mantap sekali",
        "bagus sekali",
        "hebat ya",
        "keren banget"
    ]

    implicit_negative = [
        "menguji kesabaran",
        "buang waktu",
        "bikin emosi",
        "tidak membantu",
        "ribet banget"
    ]

    has_positive = any(word in text for word in positive_words)
    has_negative = any(word in text for word in negative_words)
    has_irony = any(word in text for word in irony_markers)
    has_implicit = any(word in text for word in implicit_negative)

    if (has_positive and has_negative) or has_irony or has_implicit:
        return True

    return False

# ==========================================
# WARNA EMOSI
# ==========================================
emotion_colors = {
    "senang": "#4CAF50",
    "marah": "#F44336",
    "sedih": "#2196F3",
    "kecewa": "#FF9800",
    "netral": "#9E9E9E"
}

# ==========================================
# PREDIKSI
# ==========================================
def predict(text):

    clean_text = preprocess(text)

    inputs = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    pred_id = torch.argmax(probs, dim=1).item()

    confidence = probs[0][pred_id].item()

    emotion = id2label[pred_id]

    sarcasm = detect_sarcasm(text)

    if emotion in ["senang"]:
        sentiment = "Positif"
    elif emotion in ["marah", "sedih", "kecewa"]:
        sentiment = "Negatif"
    else:
        sentiment = "Netral"

    return emotion, sentiment, sarcasm, confidence

# ==========================================
# INPUT TEXT
# ==========================================
st.subheader("✍️ Input Teks")

user_input = st.text_area(
    "Masukkan ulasan",
    height=150,
    placeholder="Contoh: Aplikasinya bagus banget tapi sering error :)"
)

if st.button("Analisis"):

    if user_input.strip() != "":

        emotion, sentiment, sarcasm, confidence = predict(user_input)

        color = emotion_colors[emotion]

        st.markdown("## 📌 Hasil Analisis")

        st.markdown(
            f"""
            <div style="
                background-color:{color};
                padding:20px;
                border-radius:10px;
                color:white;
                text-align:center;
            ">
                <h2>{emotion.upper()}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Sentimen", sentiment)

        with col2:
            st.metric("Confidence", f"{confidence:.2%}")

        if sarcasm:
            st.error("⚠️ Terdeteksi Sarkasme")
        else:
            st.success("✅ Tidak Sarkasme")

# ==========================================
# BULK CSV
# ==========================================
st.divider()

st.subheader("📂 Upload CSV")

uploaded_file = st.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if 'content' not in df.columns:
        st.error("Kolom wajib bernama 'content'")
    else:

        emotions = []
        sentiments = []
        sarcasms = []
        confidences = []

        progress = st.progress(0)

        for i, text in enumerate(df['content']):

            emotion, sentiment, sarcasm, confidence = predict(text)

            emotions.append(emotion)
            sentiments.append(sentiment)
            sarcasms.append("Ya" if sarcasm else "Tidak")
            confidences.append(round(confidence, 4))

            progress.progress((i + 1) / len(df))

        df['emotion'] = emotions
        df['sentiment'] = sentiments
        df['sarcasm'] = sarcasms
        df['confidence'] = confidences

        st.success("✅ Analisis selesai")

        st.dataframe(df.head(20))

        # Statistik
        st.subheader("📈 Distribusi Emosi")

        emotion_counts = df['emotion'].value_counts()

        st.bar_chart(emotion_counts)

        # Download
        csv = df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="⬇️ Download Hasil",
            data=csv,
            file_name='hasil_analisis.csv',
            mime='text/csv'
        )

# ==========================================
# FOOTER
# ==========================================
st.divider()

st.caption("Developed with ❤️ using IndoBERT & Streamlit")

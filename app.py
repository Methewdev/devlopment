import os
import streamlit as st
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Analisis Emosi & Segmentasi", layout="wide")
st.title("📊 Analisis Emosi & Segmentasi Nasabah (Transformer)")

# Disable warning tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# LOAD MODEL (ANTI ERROR)
# =========================
@st.cache_resource
def load_model():
    MODEL_NAME = "envidevelopment/model2  # ⚠️ GANTI USERNAME

    try:
        with st.spinner("🔄 Loading model dari HuggingFace..."):
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

            device = torch.device("cpu")
            model.to(device)
            model.eval()

        return tokenizer, model

    except Exception as e:
        st.error("❌ Gagal load model dari HuggingFace")
        st.error(e)
        st.stop()

# Load model
tokenizer, model = load_model()

# Label mapping
id2label = model.config.id2label

# =========================
# PREDICT FUNCTION
# =========================
def predict_proba(text):
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        return {id2label[i]: float(probs[i]) for i in range(len(probs))}

    except Exception as e:
        return {"error": str(e)}

# =========================
# MENU
# =========================
menu = st.sidebar.selectbox("Menu", ["Input Teks", "Upload Dataset"])

# =========================
# INPUT TEKS
# =========================
if menu == "Input Teks":
    st.subheader("📝 Analisis Satu Ulasan")

    text = st.text_area("Masukkan teks ulasan")

    if st.button("Analisis"):
        if text.strip() == "":
            st.warning("⚠️ Masukkan teks terlebih dahulu")
        else:
            result = predict_proba(text)

            if "error" in result:
                st.error(result["error"])
            else:
                st.write("### 🔥 Probabilitas Emosi")
                st.json(result)

                label = max(result, key=result.get)
                st.success(f"🎯 Emosi Dominan: {label}")

# =========================
# UPLOAD DATASET
# =========================
elif menu == "Upload Dataset":
    st.subheader("📂 Upload Dataset CSV")

    file = st.file_uploader("Upload file CSV (kolom: text)")

    if file:
        try:
            df = pd.read_csv(file)

            if 'text' not in df.columns:
                st.error("❌ CSV harus memiliki kolom 'text'")
                st.stop()

            st.write("### Preview Data")
            st.dataframe(df.head())

            if st.button("🚀 Proses Analisis"):
                with st.spinner("⏳ Memproses data..."):

                    # Predict
                    results = df['text'].astype(str).apply(predict_proba)
                    emotion_df = pd.DataFrame(list(results))

                    if "error" in emotion_df.columns:
                        st.error("❌ Error saat prediksi")
                        st.stop()

                    df = pd.concat([df, emotion_df], axis=1)

                    # Clustering
                    emotion_cols = list(emotion_df.columns)

                    kmeans = KMeans(n_clusters=3, random_state=42)
                    df['cluster'] = kmeans.fit_predict(df[emotion_cols])

                st.success("✅ Analisis selesai!")

                # =========================
                # TAMPILKAN HASIL
                # =========================
                st.write("### 📊 Hasil Data")
                st.dataframe(df.head())

                # =========================
                # VISUALISASI CLUSTER
                # =========================
                st.write("### 📈 Distribusi Cluster")

                fig, ax = plt.subplots()
                df['cluster'].value_counts().plot(kind='bar', ax=ax)
                st.pyplot(fig)

                # =========================
                # KARAKTERISTIK CLUSTER
                # =========================
                st.write("### 🧠 Karakteristik Emosi per Cluster")
                summary = df.groupby('cluster')[emotion_cols].mean()
                st.dataframe(summary)

                # =========================
                # DOWNLOAD
                # =========================
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Hasil",
                    csv,
                    "hasil_analisis.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error("❌ Error membaca file")
            st.error(e)

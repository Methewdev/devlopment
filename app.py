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
st.set_page_config(page_title="Analisis Emosi & Segmentasi Nasabah", layout="wide")

st.title("📊 Analisis Emosi & Segmentasi Nasabah (Transformer)")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
@st.cache_resource
def load_model():
    model_path = "username/emotion-indobert"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    device = torch.device("cpu")
    model.to(device)

    return tokenizer, model

tokenizer, model = load_model()

# LABEL
id2label = model.config.id2label

# =========================
# PREDICT FUNCTION
# =========================
def predict_proba(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]

    return {id2label[i]: float(probs[i]) for i in range(len(probs))}

# =========================
# MENU
# =========================
menu = st.sidebar.selectbox("Menu", ["Input Teks", "Upload Dataset"])

# =========================
# 1. INPUT SINGLE TEXT
# =========================
if menu == "Input Teks":
    st.subheader("📝 Analisis Satu Ulasan")

    text = st.text_area("Masukkan teks ulasan")

    if st.button("Analisis"):
        if text.strip() == "":
            st.warning("Masukkan teks terlebih dahulu")
        else:
            result = predict_proba(text)

            st.write("### 🔥 Probabilitas Emosi")
            st.json(result)

            # Emosi dominan
            label = max(result, key=result.get)
            st.success(f"Emosi Dominan: {label}")

# =========================
# 2. UPLOAD DATASET
# =========================
elif menu == "Upload Dataset":
    st.subheader("📂 Upload Dataset (CSV)")

    file = st.file_uploader("Upload file CSV (kolom: text)")

    if file:
        df = pd.read_csv(file)

        if 'text' not in df.columns:
            st.error("CSV harus memiliki kolom 'text'")
        else:
            st.write("Data Preview:")
            st.dataframe(df.head())

            if st.button("Proses Analisis"):
                # =========================
                # PREDICT ALL
                # =========================
                results = df['text'].apply(predict_proba)
                emotion_df = pd.DataFrame(list(results))

                df = pd.concat([df, emotion_df], axis=1)

                # =========================
                # CLUSTERING
                # =========================
                emotion_cols = list(emotion_df.columns)

                kmeans = KMeans(n_clusters=3, random_state=42)
                df['cluster'] = kmeans.fit_predict(df[emotion_cols])

                # =========================
                # HASIL
                # =========================
                st.success("Analisis selesai!")

                st.write("### 📊 Hasil Data")
                st.dataframe(df.head())

                # =========================
                # VISUALISASI CLUSTER
                # =========================
                st.write("### 📈 Distribusi Cluster")
                cluster_counts = df['cluster'].value_counts()

                fig, ax = plt.subplots()
                cluster_counts.plot(kind='bar', ax=ax)
                st.pyplot(fig)

                # =========================
                # RATA-RATA EMOSI PER CLUSTER
                # =========================
                st.write("### 🧠 Karakteristik Cluster")
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

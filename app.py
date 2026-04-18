
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.title("Analisis Sentimen & Emosi")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("model_kamu")
    model = AutoModelForSequenceClassification.from_pretrained("model_kamu")
    return tokenizer, model

tokenizer, model = load_model()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()

    label_map = {
        0: "Negatif",
        1: "Netral",
        2: "Positif"
    }

    return label_map[pred]

# UI input
user_input = st.text_area("Masukkan komentar:")

if st.button("Prediksi"):
    if user_input:
        hasil = predict(user_input)
        st.success(f"Hasil: {hasil}")
        import pandas as pd

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if st.button("Prediksi Bulk"):
        df['hasil'] = df['ulasan'].apply(predict)
        st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download hasil", csv, "hasil.csv")
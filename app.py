import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.title("Analisis Sentimen")

@st.cache_resource
def load_model():
    model_name = "model_name = "envidevelopment/sentiment-banking"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
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

# UI
user_input = st.text_area("Masukkan komentar:")

if st.button("Prediksi"):
    if user_input:
        hasil = predict(user_input)
        st.success(f"Hasil: {hasil}")

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.title("Analisis Emosi")

@st.cache_resource
def load_model():
    MODEL_NAME = "envidevelopment/sentiment-banking"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    model.to("cpu")
    model.eval()

    return tokenizer, model

tokenizer, model = load_model()

id2label = model.config.id2label

text = st.text_area("Masukkan teks")

if st.button("Analisis"):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

    result = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    st.json(result)

import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from huggingface_hub import hf_hub_download

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

st.set_page_config(page_title="Next Word Predictor", layout="centered")


def load_assets():
    model_path = hf_hub_download(
        repo_id="Samacker25/next-word-predictor",
        repo_type="model",
        filename="next_word_predictor_model.keras",
        force_download=True
    )
    tokenizer_path = hf_hub_download(
        repo_id="Samacker25/next-word-predictor",
        repo_type="model",
        filename="tokenizer.pkl",
        force_download=True
    )

    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

model, tokenizer = load_assets()

st.title("🧠 Next Word Predictor")

text = st.text_input("Enter text")

def predict_next_word(text):
    seq = tokenizer.texts_to_sequences([text])[0]
    seq = tf.keras.preprocessing.sequence.pad_sequences([seq], maxlen=20)
    pred = np.argmax(model.predict(seq), axis=1)
    return tokenizer.index_word.get(pred[0], "Unknown")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        word = predict_next_word(text)
        st.success(f"Next word: **{word}**")

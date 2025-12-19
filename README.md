#🧠 Next Word Predictor – NLP Deep Learning Application

🔗 Live Demo

👉 Streamlit App: https://samacker25-next-word-predictor.streamlit.app/

👉 Model Hub: https://huggingface.co/Samacker25/next-word-predictor

📌 Overview

This project is an end-to-end Next Word Prediction system built using Deep Learning for NLP.
Given an input text sequence, the model predicts the most probable next word using sequence modeling.

The trained model is versioned and stored on Hugging Face Model Hub, while the inference UI is deployed using Streamlit Cloud, following clean ML deployment practices.

🚀 Features

Predicts the next word from an input text sequence

Deep learning–based NLP model (sequence modeling)

Tokenizer + model artifact separation

Hugging Face Model Hub for model registry

Streamlit-based interactive web UI

Fully reproducible & version-locked environment

🧩 Tech Stack

Language: Python

Deep Learning: TensorFlow 2.19, Keras 3

NLP: Tokenization, sequence padding, softmax prediction

Model Format: .keras (Keras v3 standard)

Model Registry: Hugging Face Model Hub

Frontend: Streamlit

Deployment: Streamlit Cloud

🏗️ Architecture
Training Notebook
      ↓
Keras (.keras) Model
      ↓
Hugging Face Model Hub
      ↓
Streamlit App (Inference)
      ↓
Live Web Application

📁 Project Structure
next-word-predictor/
├── app/
│   └── main.py
├── requirements.txt
└── README.md


Model artifacts (.keras, tokenizer.pkl) are stored separately in Hugging Face Model Hub.

⚙️ Model Loading Strategy

Training and inference environments are aligned (TensorFlow 2.19 + Keras 3)

Model is loaded dynamically from Hugging Face using hf_hub_download

Artifacts are cached safely using Streamlit resource caching

This avoids common serialization and compatibility issues.

▶️ Run Locally
pip install -r requirements.txt
streamlit run app/main.py

📌 Future Improvements

Top-K word predictions with probabilities

Transformer-based language model

FastAPI inference service

Dockerized deployment

CI/CD pipeline for model updates

👤 Author

Soumen Kundu
🔗 LinkedIn: https://www.linkedin.com/in/Samacker25

🔗 GitHub: https://github.com/Samacker25

# Streamlit App for Healthcare Chatbot
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import gdown
import re
import os

# Google Drive File IDs and Paths
file_ids = {
    "gru_model": "1VCXEiEADyVz2NLJ8b0HGU8TftpZlFSS8",
    "tokenizer": "122m9vzR4cvsRLC7lxlE8lcIfhVro_EPk",
    "label_encoder": "12C6U60REeFWUEdxbNDOIWSA_hSE0YOGV",
    "processed_data": "12MNsyrMBEylIhC__S0LPV_7cjhUTxFck",
    "data5": "11xhQufvXsTwjb4iKLp4l5S0ube8af5Rs",
    "cleaned_dataset": "11tLsqLqVF3WFLvcoTz0wcs0iYYbncMrH"
}

file_paths = {
    "gru_model": "gru_model.h5",
    "tokenizer": "tokenizer.pkl",
    "label_encoder": "label_encoder.pkl",
    "processed_data": "processed_data.csv",
    "data5": "5.csv",
    "cleaned_dataset": "cleaned_dataset_with_embeddings.pkl"
}

# Set page configuration
st.set_page_config(page_title="Smart Healthcare Chatbot", layout="wide")

# Function to download file from Google Drive
def download_file_from_gdrive(file_id, output_path):
    if not os.path.exists(output_path):
        try:
            url = f"https://drive.google.com/uc?id={file_id}&confirm=t"
            gdown.download(url, output_path, quiet=False)
        except Exception as e:
            raise RuntimeError(f"Failed to download file from Google Drive: {e}")

# Preprocess user query
def preprocess_query(query):
    corrected_query = str(TextBlob(query).correct())
    return re.sub(r'[^\w\s]', '', corrected_query.lower()).strip()

# Download required files
for key, file_id in file_ids.items():
    download_file_from_gdrive(file_id, file_paths[key])

# Load resources for Disease Q&A
@st.cache_resource
def load_disease_resources():
    # Load the dataset
    with open(file_paths["cleaned_dataset"], "rb") as f:
        df = pickle.load(f)

    # Load models
    mini_lm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    distilroberta_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    bert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    return mini_lm_model, distilroberta_model, bert_model, df

# Load resources for Medicine Recommendation
@st.cache_resource
def load_medicine_resources():
    # Load models and data
    with open(file_paths["data5"], 'rb') as f:
        data5 = pd.read_csv(file_paths["data5"])

    return data5

# Functions for Medicine Recommendation
def is_emergency(symptoms):
    emergency_symptoms = [
        "chest pain", "severe bleeding", "difficulty breathing", "sudden confusion",
        "loss of consciousness", "heart attack", "stroke", "difficulty speaking"
    ]
    for emergency in emergency_symptoms:
        if emergency in symptoms.lower():
            return True
    return False

def chatbot_response(symptoms, data5):
    if is_emergency(symptoms):
        return "This is an emergency. Please consult a healthcare professional immediately."

    # Placeholder for actual recommendation logic
    return {
        "Medicine Name": "Paracetamol",
        "Uses": "Fever and mild pain relief",
        "Side Effects": "Nausea, rash",
        "Substitutes": "Acetaminophen",
        "Therapeutic Class": "Analgesic",
        "Manufacturer": "Generic"
    }

# Load all resources
mini_lm_model, distilroberta_model, bert_model, df = load_disease_resources()
data5 = load_medicine_resources()

# Sidebar for navigation
st.sidebar.header("Navigation")
tabs = ["Disease Q&A", "Medicine Recommendation"]
selected_tab = st.sidebar.selectbox("Choose a Tab", tabs)

if selected_tab == "Disease Q&A":
    # Embedding selection for Disease Q&A
    embedding_type = st.sidebar.selectbox(
        "Choose Embedding Type:",
        ["mini_lm_embedding", "distilroberta_embedding", "bert_embedding"]
    )

    # Disease Q&A Main Interface
    st.title("🩺 Smart Healthcare Chatbot - Disease Q&A")
    user_query = st.text_input("Ask your healthcare question:", placeholder="Type your question here...")
    if user_query:
        # Preprocess query
        query_clean = preprocess_query(user_query)

        # Select model based on embedding type
        if embedding_type == "mini_lm_embedding":
            query_embedding = mini_lm_model.encode(query_clean).reshape(1, -1)
        elif embedding_type == "distilroberta_embedding":
            query_embedding = distilroberta_model.encode(query_clean).reshape(1, -1)
        elif embedding_type == "bert_embedding":
            query_embedding = bert_model.encode(query_clean).reshape(1, -1)

        # Calculate cosine similarity
        df['similarity'] = df[embedding_type].apply(
            lambda x: cosine_similarity(query_embedding, np.array(x).reshape(1, -1))[0][0]
        )
        top_match = df.loc[df['similarity'].idxmax()]

        # Display results
        st.success(f"**Answer:** {top_match['answer']}")
        st.markdown(f"**Source:** {top_match['source']}")
        st.markdown(f"**Focus Area:** {top_match['focus_area']}")

elif selected_tab == "Medicine Recommendation":
    # Medicine Recommendation Main Interface
    st.title("💊 Healthcare Medicine Recommendation Chatbot")
    user_input = st.text_input("Enter your symptoms:")
    if user_input:
        response = chatbot_response(user_input, data5)
        if isinstance(response, dict):
            st.write("### Medicine Details:")
            for key, value in response.items():
                st.write(f"**{key}:** {value}")
        else:
            st.write(response)

# Footer
st.info("💡 For accurate healthcare advice, consult a medical professional.")

# Streamlit App for Healthcare Chatbot
import streamlit as st
import pickle
import pandas as pd
import numpy as np
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
    try:
        if not os.path.exists(output_path):
            url = f"https://drive.google.com/uc?id={file_id}&confirm=t"
            gdown.download(url, output_path, quiet=False)
    except Exception as e:
        st.error(f"Failed to download file: {output_path}. Please ensure it exists.")
        raise e

# Preprocess user query
def preprocess_query(query):
    corrected_query = str(TextBlob(query).correct())
    return re.sub(r'[^\w\s]', '', corrected_query.lower()).strip()

# Load resources for Disease Q&A
@st.cache_resource
def load_disease_resources():
    download_file_from_gdrive(file_ids["cleaned_dataset"], file_paths["cleaned_dataset"])
    with open(file_paths["cleaned_dataset"], "rb") as f:
        df = pickle.load(f)
    mini_lm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    distilroberta_model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    bert_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    return mini_lm_model, distilroberta_model, bert_model, df

# Load resources for Medicine Recommendation
@st.cache_resource
def load_medicine_resources():
    download_file_from_gdrive(file_ids["data5"], file_paths["data5"])
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

def recommend_medicine(symptoms, data5):
    if is_emergency(symptoms):
        return "This is an emergency. Please consult a healthcare professional immediately."
    
    # Convert user symptoms to lowercase
    symptoms = symptoms.lower()
    
    # Check if symptoms match any entries in the dataset
    matches = data5[data5['Symptoms'].str.contains(symptoms, case=False, na=False)]
    
    if not matches.empty:
        # Return the first match or all matches
        return matches.iloc[0].to_dict()  # Return as a dictionary
    else:
        return "No suitable medicine found for the given symptoms. Please consult a healthcare professional."

# Sidebar for navigation
st.sidebar.header("Navigation")
tabs = ["Disease Q&A", "Medicine Recommendation"]
selected_tab = st.sidebar.selectbox("Choose a Tab", tabs)

if selected_tab == "Disease Q&A":
    # Lazy loading for Disease Q&A
    st.info("Loading Disease Q&A resources...")
    mini_lm_model, distilroberta_model, bert_model, df = load_disease_resources()
    embedding_type = st.sidebar.selectbox(
        "Choose Embedding Type:",
        ["mini_lm_embedding", "distilroberta_embedding", "bert_embedding"]
    )
    st.title("🩺 Smart Healthcare Chatbot - Disease Q&A")
    user_query = st.text_input("Ask your healthcare question:", placeholder="Type your question here...")
    if user_query:
        query_clean = preprocess_query(user_query)
        if embedding_type == "mini_lm_embedding":
            query_embedding = mini_lm_model.encode(query_clean).reshape(1, -1)
        elif embedding_type == "distilroberta_embedding":
            query_embedding = distilroberta_model.encode(query_clean).reshape(1, -1)
        elif embedding_type == "bert_embedding":
            query_embedding = bert_model.encode(query_clean).reshape(1, -1)
        df['similarity'] = df[embedding_type].apply(
            lambda x: cosine_similarity(query_embedding, np.array(x).reshape(1, -1))[0][0]
        )
        top_match = df.loc[df['similarity'].idxmax()]
        st.success(f"**Answer:** {top_match['answer']}")
        st.markdown(f"**Source:** {top_match['source']}")
        st.markdown(f"**Focus Area:** {top_match['focus_area']}")

elif selected_tab == "Medicine Recommendation":
    # Lazy loading for Medicine Recommendation
    st.info("Loading Medicine Recommendation resources...")
    data5 = load_medicine_resources()
    st.title("💊 Healthcare Medicine Recommendation Chatbot")
    user_input = st.text_input("Enter your symptoms:")
    if user_input:
        response = recommend_medicine(user_input, data5)
        if isinstance(response, dict):
            st.write("### Medicine Details:")
            for key, value in response.items():
                st.write(f"**{key}:** {value}")
        else:
            st.warning(response)

# Footer
st.info("💡 For accurate healthcare advice, consult a medical professional.")

import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import process
import gdown
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Smart Healthcare Chatbot", layout="wide")

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

# Function to download files from Google Drive
def download_file(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

# Download all required files
for key, file_id in file_ids.items():
    download_file(file_id, file_paths[key])

# Load Models and Data
gru_model = load_model(file_paths["gru_model"])
tokenizer = joblib.load(file_paths["tokenizer"])
label_encoder = joblib.load(file_paths["label_encoder"])
processed_data = pd.read_csv(file_paths["processed_data"])
data5 = pd.read_csv(file_paths["data5"])
mini_lm_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Emergency Symptoms
emergency_symptoms = [
    "chest pain", "severe bleeding", "difficulty breathing", "sudden confusion",
    "weakness or numbness on one side", "loss of consciousness", "severe headache",
    "seizures", "severe burns", "uncontrolled vomiting", "high fever", "persistent dizziness",
    "major trauma", "heart attack", "stroke", "difficulty speaking", "severe allergic reaction",
    "intense abdominal pain", "continuous chest pressure", "sudden vision loss",
    "high or low blood sugar", "severe dehydration", "painful swelling", "sudden severe back pain",
    "persistent vomiting with blood", "sepsis symptoms", "head trauma", "difficulty walking"
]

# Function to check for emergency scenarios
def is_emergency(symptoms):
    for emergency in emergency_symptoms:
        if emergency in symptoms.lower():
            return True
    return False

# Medicine Details Functionality
def get_medicine_details(medicine_name):
    corrected_name = correct_medicine_name(medicine_name)
    if corrected_name:
        details = processed_data[processed_data['medicine_name'] == corrected_name].iloc[0]
        return {
            "Medicine Name": details['medicine_name'],
            "Uses": details['combined_uses'],
            "Side Effects": details['combined_side_effects'],
            "Substitutes": details['combined_substitutes'],
            "Therapeutic Class": details['therapeutic_class'],
            "Image URL": details['image_url'],
            "Manufacturer": details['manufacturer']
        }
    else:
        return {"Error": "Medicine not found. Please check the spelling or try another name."}

# Medicine Recommendation Functionality
def recommend_medicine(symptoms):
    if is_emergency(symptoms):
        return "This is an emergency. Please consult a healthcare professional immediately."
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data5['uses'])
    user_vector = vectorizer.transform([symptoms])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)
    top_indices = similarity_scores.argsort()[0][-3:][::-1]
    return data5.iloc[top_indices][['name', 'uses']]

# Disease Q&A Functionality
def answer_disease_query(query):
    with open(file_paths["cleaned_dataset"], "rb") as f:
        df = pickle.load(f)

    query_embedding = mini_lm_model.encode(query).reshape(1, -1)
    df['similarity'] = df['mini_lm_embedding'].apply(
        lambda x: cosine_similarity(query_embedding, np.array(x).reshape(1, -1))[0][0]
    )
    top_match = df.loc[df['similarity'].idxmax()]
    return {
        "Answer": top_match['answer'],
        "Source": top_match['source'],
        "Focus Area": top_match['focus_area']
    }

# Streamlit App
st.title("Smart Healthcare Chatbot")

# Tabs for navigation
tab1, tab2, tab3 = st.tabs(["Disease Q&A", "Medicine Recommendation", "Medicine Details"])

# Tab 1: Disease Q&A
with tab1:
    st.header("🩺 Disease Q&A Chatbot")
    user_input = st.text_input("Ask a healthcare question:", "")
    if user_input:
        response = answer_disease_query(user_input)
        st.write(f"**Answer:** {response['Answer']}")
        st.write(f"**Source:** {response['Source']}")
        st.write(f"**Focus Area:** {response['Focus Area']}")

# Tab 2: Medicine Recommendation
with tab2:
    st.header("💊 Medicine Recommendation Chatbot")
    user_input = st.text_input("Enter your symptoms:", "")
    if user_input:
        recommendations = recommend_medicine(user_input)
        if isinstance(recommendations, str):  # Emergency response
            st.write(recommendations)
        else:
            st.write("### Recommended Medicines:")
            st.write(recommendations)

# Tab 3: Medicine Details
with tab3:
    st.header("🔍 Medicine Details Lookup")
    user_input = st.text_input("Enter Medicine Name:", "")
    if user_input:
        details = get_medicine_details(user_input)
        if "Error" in details:
            st.write(details["Error"])
        else:
            st.write("### Medicine Details:")
            for key, value in details.items():
                if key == "Image URL" and value != "Not Available":
                    st.image(value, caption="Medicine Image")
                else:
                    st.write(f"**{key}:** {value}")

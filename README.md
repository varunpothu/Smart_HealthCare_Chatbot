# 🏥 **Smart Healthcare Chatbot** for Personalized Health Condition Analysis

## 🚀 Overview
The **Smart Healthcare Chatbot** is an AI-driven system that provides personalized health condition analysis, symptom checking, virtual consultations, medication reminders, and health education. It offers real-time support for general health inquiries and can assist during emergencies by suggesting nearby NHS facilities using the user’s location. The chatbot integrates Machine Learning (ML), Natural Language Processing (NLP) and Large Language Model (LLM) techniques to analyze user inputs and give tailored recommendations.

![Smart_HealthCare_Chatbot](https://drive.google.com/uc?id=1e9l881IAC7H3flLam8fjbGiC9dGfiv_5)

## 📑 Table of Contents
1. [Features](#-features)
2. [Technology Stack](#-technology-stack)
3. [Installation](#-installation)
4. [Usage](#-usage)
5. [Datasets](#-datasets)
6. [API Integration](#-api-integration)
7. [Project Structure](#-project-structure)
8. [License & Attribution](#-license--attribution)
9. [Contact Information](#-contact-information)

## ✨ Features
- **🤖 Virtual Consultation**: Users can input symptoms or health concerns to receive condition analysis and next-step recommendations.
- **🩺 Symptom Checker**: Helps users analyze symptoms and suggests potential health conditions.
- **💊 Medication Reminders**: Users can set reminders for taking medications based on doctor prescriptions.
- **📚 Health Education**: Offers information about health conditions, medicines, and general wellness.
- **🚨 Emergency Assistance**: Detects emergency conditions and locates nearby NHS facilities using real-time location data.
- **🔍 Personalized Nutrition Suggestions**: Recommends appropriate nutrition based on health conditions.
- **🧑‍⚕️ Doctor and Hospital Recommendations**: Suggests the exact type of doctor and nearest hospitals for consultation based on symptoms.
- **📍 Location-Based Assistance**: Utilizes the user's current location to show nearby medical facilities and doctors.

## 🛠️ Technology Stack
- **Language**: Python 🐍
- **Web Framework**: Flask 🌐
- **Machine Learning & NLP**: Scikit-learn, TensorFlow, NLTK, SpaCy, BERT 🤖
- **Data Visualization**: Power BI 📊
- **APIs**: Google Maps API 🗺️ for geolocation and nearby hospital suggestions
- **Storage**: Encrypted CSV for user conversation data
- **Deployment**: Flask (for web interface), Docker (for containerization)

## 🛠️ Installation
To get started with this project locally, follow the steps below:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/smart-healthcare-chatbot.git
   ```

2. **Navigate to the project directory**:
   ```bash
   cd smart-healthcare-chatbot
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (e.g., for Google Maps API):
   - Create a `.env` file with your API keys and environment configurations.
   ```bash
   GOOGLE_MAPS_API_KEY=your-api-key
   ```

5. **Run the Flask server**:
   ```bash
   python app.py
   ```

6. **Access the chatbot interface**:
   Open your browser and navigate to `http://localhost:5000`.

## 🚀 Usage
Once the Flask server is running, you can interact with the chatbot by entering health symptoms or concerns. The chatbot provides:
- Symptom analysis and potential health conditions
- Medication information, including usage, side effects, and substitutes
- Reminders for taking prescribed medications
- Emergency assistance by showing nearby NHS facilities

![Smart_HealthCare_Chatbot1](https://drive.google.com/uc?id=1pHuY-uZXCOfoc19Oqq3dIqeUAih2QFWI)

## 📂 Datasets
This project utilizes various datasets to provide accurate health recommendations. The datasets are hosted on Kaggle and are subject to different licenses. Ensure you review and comply with the dataset licenses.

### Datasets:
1. **[Medicine Recommendation System Dataset](https://www.kaggle.com/datasets/noorsaeed/medicine-recommendation-system-dataset/data)**  
   License: **Apache 2.0**
   
2. **[11000 Medicine Details Dataset](https://www.kaggle.com/datasets/singhnavjot2062001/11000-medicine-details)**  
   License: **CC0: Public Domain**

3. **[WebMD Drug Reviews Dataset](https://www.kaggle.com/datasets/rohanharode07/webmd-drug-reviews-dataset/data)**  
   License: **CC0: Public Domain**

4. **[Drugs Side Effects and Medical Condition Dataset](https://www.kaggle.com/datasets/jithinanievarghese/drugs-side-effects-and-medical-condition)**  
   License: **CC0: Public Domain**

5. **[MedQuAD Medical Question Answer Dataset](https://www.kaggle.com/datasets/pythonafroz/medquad-medical-question-answer-for-ai-research/data)**  
   License: **CC BY-SA 4.0**

6. **[LayoutLM Dataset](https://www.kaggle.com/datasets/jpmiller/layoutlm/data)**  
   License: **CC BY-SA 4.0**

7. **[250k Medicines Usage, Side Effects, and Substitutes Dataset](https://www.kaggle.com/datasets/shudhanshusingh/250k-medicines-usage-side-effects-and-substitutes)**  
   License: **CC BY-SA 4.0**

## 🌍 API Integration
- **Google Maps API**: For real-time location tracking and suggesting nearby hospitals or emergency centers based on the user's current location.
- **NHS Facility Data**: Integrated to provide real-time suggestions for NHS facilities.

## 📁 Project Structure
```bash
smart-healthcare-chatbot/
│
├── data/                          # Datasets folder
├── models/                        # Pre-trained machine learning models
├── static/                        # Static files (images, CSS)
├── templates/                     # HTML files for the web front-end
├── app.py                         # Flask application logic
├── requirements.txt               # Dependencies
├── README.md                      # Project documentation
├── .env                           # Environment variables for API keys
└── utils/                         # Utility scripts for data processing
```

## 📄 License & Attribution
This Smart Healthcare Chatbot for Personalized Health Condition Analysis project is part of my final project for academic purposes. It utilizes various publicly available datasets that are licensed under CC0: Public Domain, CC BY-SA 4.0, and Apache 2.0 licenses. Proper compliance with these licenses is mandatory when using, modifying, or distributing this project.

### Project License:
Since this project integrates multiple open datasets, it does not follow a single unified license but rather adheres to the terms set by the specific datasets used.

### Dataset Licenses:
- **CC0: Public Domain**: No attribution required.
- **CC BY-SA 4.0**: Attribution required, and any derivatives must be shared under the same license.
- **Apache 2.0**: Free to use for commercial purposes with proper attribution.

## 📧 Contact Information
If you have any questions or need further assistance, feel free to reach out:
- **Email**: pothuvarun@gmail.com
- **GitHub**: [My GitHub Profile](https://github.com/varunpothu)
  
Academic Usage:
This project is submitted as part of my final academic project. If you are using or adapting this project for academic purposes, please ensure that appropriate attribution is provided to both the original dataset authors and this project.


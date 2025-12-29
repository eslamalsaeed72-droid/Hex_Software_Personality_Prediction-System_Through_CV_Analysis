import streamlit as st
import joblib
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pdfplumber
import plotly.express as px
import pandas as pd

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)

# Initialize NLTK
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Clean and preprocess input text"""
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = text.replace('|||', ' ')
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words 
             if word not in stop_words and len(word) > 2]
    return ' '.join(words)

# Load models and components
@st.cache_resource
def load_models():
    model_mbti = joblib.load('model_mbti.pkl')
    model_big5 = joblib.load('model_big5.pkl')
    vectorizer_mbti = joblib.load('vectorizer_mbti.pkl')
    vectorizer_big5 = joblib.load('vectorizer_big5.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    with open('mbti_to_big5_mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)
    return model_mbti, model_big5, vectorizer_mbti, vectorizer_big5, label_encoder, mapping

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

# Prediction function
def predict_personality(text, model_mbti, model_big5, vectorizer_mbti, vectorizer_big5, label_encoder, mapping):
    cleaned = clean_text(text)
    if len(cleaned.split()) < 10:
        return None, "Warning: Extracted text is too short for reliable prediction."

    X_mbti = vectorizer_mbti.transform([cleaned])
    mbti_pred = model_mbti.predict(X_mbti)[0]
    mbti_type = label_encoder.inverse_transform([mbti_pred])[0]

    mapped = mapping.get(mbti_type, {'O':0.5,'C':0.5,'E':0.5,'A':0.5,'N':0.5})

    X_big5 = vectorizer_big5.transform([cleaned])
    big5_probs = model_big5.predict_proba(X_big5)
    direct = {}
    traits = ['O', 'C', 'E', 'A', 'N']
    for i, t in enumerate(traits):
        direct[t] = big5_probs[i][0][1]

    final = {}
    for t in traits:
        final[t] = round((0.6 * direct[t] + 0.4 * mapped[t]) * 100, 1)

    return {
        'predicted_mbti': mbti_type,
        'Openness': final['O'],
        'Conscientiousness': final['C'],
        'Extraversion': final['E'],
        'Agreeableness': final['A'],
        'Neuroticism': final['N']
    }, None

# Streamlit UI
st.set_page_config(page_title="Personality Prediction from CV", layout="centered")
st.title("🧠 Personality Prediction System")
st.markdown("### Analyze personality traits from any resume or CV (PDF) using AI")

# Load models
with st.spinner("Loading models..."):
    model_mbti, model_big5, vectorizer_mbti, vectorizer_big5, label_encoder, mapping = load_models()

st.success("Models loaded successfully!")

uploaded_file = st.file_uploader("Upload your CV (PDF format)", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting text and analyzing personality..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        
        if len(raw_text.strip()) < 100:
            st.error("Could not extract sufficient text from the PDF. Please ensure it's a text-based PDF.")
        else:
            st.success("Text extracted successfully!")
            st.markdown("#### Preview of extracted text:")
            st.text(raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text)
            
            result, error = predict_personality(
                raw_text, model_mbti, model_big5, 
                vectorizer_mbti, vectorizer_big5, label_encoder, mapping
            )
            
            if error:
                st.warning(error)
            else:
                st.markdown("## 🎯 Personality Analysis Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted MBTI Type", result['predicted_mbti'])
                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)  # spacing
                
                # Radar chart
                traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
                values = [result[t] for t in traits]
                
                df_chart = pd.DataFrame(dict(
                    r=values,
                    theta=traits
                ))
                
                fig = px.line_polar(df_chart, r='r', theta='theta', line_close=True,
                                    range_r=[0,100],
                                    title="Big Five Personality Traits")
                fig.update_traces(fill='toself')
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Individual scores
                st.markdown("#### Detailed Scores")
                cols = st.columns(5)
                for i, trait in enumerate(traits):
                    with cols[i]:
                        st.metric(trait, f"{result[trait]}%")
                
                st.info("Note: This is an AI-based estimation for research/entertainment purposes. Not a substitute for professional assessment.")

st.markdown("---")
st.caption("Built with ❤️ using Machine Learning | MBTI + Big Five Ensemble Model")

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
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import urllib.parse

# NLTK Setup
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('punkt', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = text.replace('|||', ' ')
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return ' '.join(words)

# Load Models
@st.cache_resource
def load_models():
    model_mbti = joblib.load('models/model_mbti.pkl')
    model_big5 = joblib.load('models/model_big5.pkl')
    vectorizer_mbti = joblib.load('models/vectorizer_mbti.pkl')
    vectorizer_big5 = joblib.load('models/vectorizer_big5.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    with open('models/mbti_to_big5_mapping.pkl', 'rb') as f:
        mapping = pickle.load(f)
    return model_mbti, model_big5, vectorizer_mbti, vectorizer_big5, label_encoder, mapping

# PDF Text Extraction
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

# Prediction
def predict_personality(text, model_mbti, model_big5, vectorizer_mbti, vectorizer_big5, label_encoder, mapping):
    cleaned = clean_text(text)
    if len(cleaned.split()) < 10:
        return None, "Text too short for reliable analysis."

    X_mbti = vectorizer_mbti.transform([cleaned])
    mbti_pred = model_mbti.predict(X_mbti)[0]
    mbti_type = label_encoder.inverse_transform([mbti_pred])[0]

    mapped = mapping.get(mbti_type, {'O':0.5,'C':0.5,'E':0.5,'A':0.5,'N':0.5})

    X_big5 = vectorizer_big5.transform([cleaned])
    big5_probs = model_big5.predict_proba(X_big5)
    direct = {t: big5_probs[i][0][1] for i, t in enumerate(['O', 'C', 'E', 'A', 'N'])}

    final = {t: round((0.6 * direct[t] + 0.4 * mapped[t]) * 100, 1) for t in direct}

    return {
        'predicted_mbti': mbti_type,
        'Openness': final['O'],
        'Conscientiousness': final['C'],
        'Extraversion': final['E'],
        'Agreeableness': final['A'],
        'Neuroticism': final['N']
    }, None

# PDF Report
def generate_pdf_report(result):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height - 80, "Personality Analysis Report")
    c.setFont("Helvetica", 14)
    c.drawString(100, height - 140, f"MBTI Type: {result['predicted_mbti']}")
    y = height - 180
    for trait in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']:
        c.drawString(100, y, f"{trait}: {result[trait]}%")
        y -= 30
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(100, y - 40, "AI-generated estimation for research purposes only.")
    c.save()
    buffer.seek(0)
    return buffer

# Streamlit App
st.set_page_config(page_title="CV Personality Analyzer", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {background-color: #0077b5; color: white; border-radius: 8px;}
    .linkedin-btn {background-color: #0077b5 !important;}
    </style>
""", unsafe_allow_html=True)

st.title("🧠 AI-Powered Personality Analysis from CV")
st.markdown("Upload your resume to discover your estimated Big Five personality traits using advanced machine learning.")

with st.sidebar:
    st.header("🔗 Quick Actions")
    if st.button("📄 Import from LinkedIn (Download PDF)", use_container_width=True):
        st.markdown("""
        ### How to import from LinkedIn:
        1. Go to [LinkedIn Data Export](https://www.linkedin.com/psettings/member-data)
        2. Request your data archive
        3. Download your **Resume** or **Profile** as PDF
        4. Upload it here!
        """)
        st.info("LinkedIn currently restricts direct API access. Manual export is the official method.")
    
    st.markdown("---")
    st.caption("Built with ❤️ for HR Tech Innovation")

# Load models
with st.spinner("Loading AI models..."):
    model_mbti, model_big5, vectorizer_mbti, vectorizer_big5, label_encoder, mapping = load_models()
st.success("Ready for analysis!")

uploaded_file = st.file_uploader("Upload your CV (PDF)", type="pdf")

if uploaded_file:
    with st.spinner("Processing your CV..."):
        raw_text = extract_text_from_pdf(uploaded_file)
        if len(raw_text.strip()) < 100:
            st.error("Not enough text extracted. Please use a text-based PDF.")
        else:
            st.success("Analysis complete!")
            result, error = predict_personality(raw_text, model_mbti, model_big5, vectorizer_mbti, vectorizer_big5, label_encoder, mapping)
            
            if result:
                st.metric("Predicted MBTI Type", result['predicted_mbti'])
                
                # Radar Chart
                traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
                fig = px.line_polar(
                    r=[result[t] for t in traits],
                    theta=traits,
                    line_close=True,
                    range_r=[0,100],
                    title="Your Personality Profile"
                )
                fig.update_traces(fill='toself', line_color='#0077b5')
                st.plotly_chart(fig, use_container_width=True)
                
                # Scores
                cols = st.columns(5)
                for col, trait in zip(cols, traits):
                    with col:
                        st.metric(trait, f"{result[trait]}%")
                
                # Actions
                col1, col2 = st.columns(2)
                with col1:
                    pdf_buffer = generate_pdf_report(result)
                    st.download_button("📄 Download Report", pdf_buffer, "personality_report.pdf", "application/pdf")
                
                with col2:
                    share_text = f"I just analyzed my personality traits using AI! I'm {result['predicted_mbti']} with {result['Openness']}% Openness. Try it yourself!"
                    share_url = f"https://www.linkedin.com/sharing/share-offsite/?url={urllib.parse.quote(st.secrets.get('APP_URL', 'https://your-app-link.streamlit.app'))}"
                    st.markdown(f'<a href="{share_url}" target="_blank"><button style="background:#0077b5;color:white;padding:10px;border:none;border-radius:8px;cursor:pointer;">🔗 Share on LinkedIn</button></a>', unsafe_allow_html=True)

st.info("This tool is for educational and research purposes. Results are AI estimates only.")

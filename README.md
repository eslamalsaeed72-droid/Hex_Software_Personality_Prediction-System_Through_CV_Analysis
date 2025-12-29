```markdown
# AI-Powered Personality Prediction from CV Analysis

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.6-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Deployed-success)

## Project Overview

**Personality Prediction System** is an advanced AI tool that analyzes the textual content of a resume or CV to estimate the candidate's **Big Five (OCEAN) personality traits**:

- **Openness** – Creativity and willingness to experience new things
- **Conscientiousness** – Organization, responsibility, and dependability
- **Extraversion** – Sociability, assertiveness, and energy
- **Agreeableness** – Cooperation, compassion, and trust
- **Neuroticism** – Emotional stability and stress response

The system uses a **dual-model ensemble approach** combining MBTI classification with direct Big Five prediction to deliver robust and insightful results. It is designed as a **proof-of-concept for responsible AI in talent assessment**, providing supplementary insights to support — not replace — human judgment in recruitment and personal development.

> **Ethical Note**: This tool generates probabilistic estimates based on linguistic patterns. Results are for educational, research, and exploratory purposes only and should never be used as the sole basis for employment decisions.

## Live Demo

[Try the Application]: https://2kactw6jnffgnhocrqkmss.streamlit.app/ 

## Key Features

- **PDF Resume Upload** with automatic text extraction
- **Interactive Radar Chart** visualizing the Big Five personality profile
- **Predicted MBTI Type** as complementary insight
- **Detailed Trait Breakdown** with clear explanations
- **Downloadable PDF Report** for sharing or archiving results
- **LinkedIn Integration** – Guidance for importing profile data and sharing results
- **Modern, Responsive UI** built with Streamlit

## How It Works

1. **MBTI Classification Model**  
   Trained on ~8,675 labeled social media posts to predict one of 16 MBTI types.

2. **Big Five Direct Prediction Model**  
   Trained on 1,578 personality-labeled essays using multi-output classification.

3. **Ensemble Fusion**  
   Combines direct trait probabilities (60%) with psychologically validated MBTI-to-Big Five mapping (40%) for balanced, reliable scores.

## Technologies Used

| Category               | Technology                          |
|------------------------|-------------------------------------|
| Programming Language   | Python 3.10+                        |
| Machine Learning       | scikit-learn                        |
| Text Processing        | NLTK, regex                         |
| Feature Engineering    | TF-IDF Vectorization                |
| PDF Handling           | pdfplumber                          |
| Frontend & Deployment  | Streamlit                           |
| Visualization          | Plotly (Interactive Radar Chart)    |
| Report Generation      | ReportLab                           |
| Model Persistence      | joblib, pickle                      |

## Datasets

- **MBTI Dataset** – [Kaggle MBTI Type Dataset](https://www.kaggle.com/datasets/datasnaek/mbti-type) (~8,675 samples)
- **Big Five Essays** – [Hugging Face essays-big5](https://huggingface.co/datasets/jingjietan/essays-big5) (1,578 labeled essays)

## Project Structure

```
cv-personality-predictor/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Project dependencies
├── README.md                       # Project documentation
└── models/                         # Trained models and components
    ├── model_mbti.pkl
    ├── model_big5.pkl
    ├── vectorizer_mbti.pkl
    ├── vectorizer_big5.pkl
    ├── label_encoder.pkl
    └── mbti_to_big5_mapping.pkl
```

## Local Installation & Running

```bash
git clone https://github.com/yourusername/cv-personality-predictor.git
cd cv-personality-predictor

python -m venv venv
source venv/bin/activate          # On Windows: venv\Scripts\activate

pip install -r requirements.txt

streamlit run app.py
```

## Deployment

The application is deployed on **Streamlit Community Cloud** (free tier):
1. Push repository to GitHub
2. Connect GitHub account on [share.streamlit.io](https://share.streamlit.io)
3. Create new app → Select repository → Main file: `app.py`

## Responsible & Ethical Use

- Personality inference from text has inherent limitations and potential biases
- Results may vary across languages, cultures, and writing styles
- Always protect user privacy and comply with data protection regulations
- Use only as a supplementary tool in holistic evaluation processes

## Future Enhancements

- Support for DOCX and multilingual resumes
- Integration of transformer-based models (e.g., BERT) for higher accuracy
- Confidence scoring and textual explanation highlights
- Comparison mode against job description personality requirements
- API endpoint for enterprise integration

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

## Author

Eslam Alsaeed 
Machine Learning Engineer | AI & HR Technology Enthusiast

- LinkedIn: [linkedin.com/in/eslam-alsaeed-1a23921aa]
- GitHub: [https://github.com/eslamalsaeed72-droid]

---

**A research-driven exploration of responsible AI applications in talent analytics and personal development.**.

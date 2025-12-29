# Personality Prediction System from CV Analysis

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.6-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Project-Complete-success)

## Project Overview

**Personality Prediction System Through CV Analysis** is an AI-powered tool designed to estimate candidates' Big Five personality traits (OCEAN) from their resume or CV content using machine learning.

The system analyzes textual elements such as word choices, professional experiences, achievements, and writing style to predict five core personality dimensions:
- **Openness**
- **Conscientiousness**
- **Extraversion**
- **Agreeableness**
- **Neuroticism**

This tool aims to support talent acquisition processes by providing data-driven insights into candidate personality profiles, helping reduce unconscious bias and enhance objective evaluation when used responsibly alongside traditional assessment methods.

> **Important Note**: This system provides probabilistic estimates based on text patterns and should **never** replace professional psychological assessments or human judgment in hiring decisions.

## Key Features

- **PDF Resume Upload & Text Extraction** – Supports standard text-based CVs
- **Dual-Model Ensemble Approach** – Combines MBTI classification with direct Big Five prediction for improved robustness
- **Interactive Radar Chart Visualization** – Clear, intuitive display of personality trait scores
- **Predicted MBTI Type** – Additional insight from the Myers-Briggs classification
- **Responsive Web Interface** – Built with Streamlit for seamless user experience

## Demo

Live Demo:(https://2kactw6jnffgnhocrqkmss.streamlit.app/)

## How It Works (Technical Architecture)

The system employs a **two-stage ensemble model**:

1. **MBTI Classification Model**
   - Trained on ~8,675 social media posts labeled with 16 MBTI types
   - Uses TF-IDF features with Logistic Regression
   - Predicts most likely MBTI personality type from CV text

2. **Big Five Direct Prediction Model**
   - Trained on 1,578 personality-labeled essays
   - Multi-output binary classification for each OCEAN trait
   - Uses TF-IDF + Logistic Regression

3. **Ensemble Fusion**
   - Maps predicted MBTI type to approximate Big Five scores using established psychological correlations
   - Combines direct Big Five probabilities (60%) with MBTI-mapped scores (40%)
   - Outputs final trait percentages (0–100%)

## Technologies Used

| Component              | Technology                          |
|-----------------------|-------------------------------------|
| Language              | Python 3.10+                        |
| Machine Learning      | scikit-learn                        |
| Text Processing       | NLTK, regex                         |
| Feature Extraction    | TF-IDF Vectorizer                   |
| PDF Processing        | pdfplumber                          |
| Web Framework         | Streamlit                           |
| Visualization         | Plotly (Radar Chart)                |
| Model Persistence     | joblib, pickle                      |
| Deployment            | Streamlit Community Cloud           |

## Dataset Sources

- **MBTI Dataset**: [Kaggle - MBTI Type](https://www.kaggle.com/datasets/datasnaek/mbti-type) (~8,675 samples)
- **Big Five Essays Dataset**: [Hugging Face - essays-big5](https://huggingface.co/datasets/jingjietan/essays-big5) (1,578 labeled essays)

## Installation & Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/cv-personality-predictor.git
cd cv-personality-predictor


# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py
```

## File Structure

```
Hex_Software_Personality_Prediction-System_Through_CV_Analysis/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
└── models/                         # Trained models and components
├── model_mbti.pkl
├── model_big5.pkl
├── vectorizer_mbti.pkl
├── vectorizer_big5.pkl
├── label_encoder.pkl
└── mbti_to_big5_mapping.pkl
```

## Usage Guidelines

1. Upload a **text-based PDF resume** (scanned/image-only PDFs may not work properly)
2. Wait for text extraction and analysis
3. View predicted MBTI type and Big Five trait percentages
4. Interpret results as supplementary insights only

## Ethical Considerations

- This tool analyzes language patterns and should not be used as the sole basis for employment decisions
- Personality prediction from text has limitations and potential cultural/language biases
- Always combine with structured interviews, references, and skills assessments
- Respect candidate privacy and data protection regulations (GDPR, CCPA, etc.)

## Future Improvements

- Support for DOCX and other formats
- Multilingual capabilities
- Fine-tuning with domain-specific HR/resume datasets
- Integration of transformer-based models (e.g., BERT) for higher accuracy
- Confidence scoring and explanation highlights

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Eslam Alsaeed
Machine Learning Enthusiast | AI for HR Innovation

---

**Built for educational and research purposes to explore AI applications in talent assessment.**

For questions or contributions, please open an issue or submit a pull request.

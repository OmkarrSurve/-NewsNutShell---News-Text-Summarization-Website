# 📰 NewsNutshell — News Text Summarization Web Application
 
> A real-time news summarization platform that condenses lengthy news articles into concise, meaningful summaries using both **Abstractive (BART-large)** and **Extractive (KNN + TF-IDF)** NLP techniques.
 
---

 
## 📌 Table of Contents
 
- [About the Project](#about-the-project)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Summarization Approaches](#summarization-approaches)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Challenges & Limitations](#challenges--limitations)
- [Future Work](#future-work)
- [References](#references)
---
 
## 📖 About the Project
 
**NewsNutshell** addresses the problem of information overload in today's fast-paced world by providing concise, high-quality summaries of current news events in real time.
 
The platform fetches live articles via the **News API**, leverages the **CNN/DailyMail dataset** for training, and delivers two types of summaries:
- **Abstractive** — rephrases and synthesizes content using the BART-large transformer model.
- **Extractive** — selects the most important sentences from the original article using KNN and TF-IDF scoring.
Users get a clean, interactive web interface to choose their summarization preferences, view history, and provide feedback.
 
---
 
## ✨ Features
 
- 🔐 **User Authentication** — Register/Login as General User or Admin
- 📝 **Dual Summarization Modes** — Abstractive (BART-large) and Extractive (KNN)
- 🎛️ **Custom Summary Settings** — Choose number of sentences, keywords, and output format (paragraph or bullet points)
- 📡 **Real-Time News Feed** — Live CNN news articles fetched via News API
- 🕓 **Summarization History** — View past summaries per user
- 📊 **Admin Panel** — View all users' summarization history and feedback
- 💬 **Feedback System** — Users can submit feedback on summary quality
- 🔑 **Keyword Extraction** — Highlights key terms from the summary
- 📐 **Cosine Similarity Score** — Measures how closely the summary reflects the original text
---
 
## 🛠️ Tech Stack
 
### Frontend
- HTML5, CSS3, JavaScript
### Backend
- Python, Flask
### NLP & Machine Learning
| Library | Purpose |
|---|---|
| `transformers` (HuggingFace) | BART-large model for abstractive summarization |
| `torch` (PyTorch) | Model training and inference |
| `scikit-learn` | KNN, TF-IDF vectorization |
| `nltk` | Tokenization, stopword removal |
| `spaCy` | NLP pipeline, NER, lemmatization |
| `rouge-score` | ROUGE evaluation metrics |
| `numpy` & `pandas` | Data handling and preprocessing |
| `matplotlib` & `seaborn` | Visualization |
 
### Data & APIs
- **Hugging Face `datasets`** — CNN/DailyMail dataset
- **News API** — Real-time article fetching
- **BeautifulSoup / Scrapy** — Web scraping support
---
 
## 🏗️ Project Architecture
 
```
User Input (Text / News Article)
        │
        ▼
   Flask Backend
        │
   ┌────┴────┐
   │         │
   ▼         ▼
Abstractive  Extractive
(BART-large) (KNN + TF-IDF)
   │         │
   └────┬────┘
        │
   Post-processing
  (Keywords, Cosine Similarity, Formatting)
        │
        ▼
   Frontend Output
  (Summary + Metrics)
```
 
---
 
## 📦 Dataset
 
- **Source:** [CNN/DailyMail Dataset](https://huggingface.co/datasets/cnn_dailymail) via Hugging Face
- **Size:** 35,000 article–summary pairs
- **Split:** 80% Training (28,000) / 10% Validation (3,500) / 10% Test (3,500)
```python
from datasets import load_dataset
 
dataset = load_dataset("cnn_dailymail", "3.0.0")
train_data = dataset['train']
val_data   = dataset['validation']
test_data  = dataset['test']
```
 
---
 
## 🤖 Summarization Approaches
 
### 1. Abstractive Summarization — BART-large
 
- **Model:** `facebook/bart-large-cnn`
- **Technique:** Fine-tuned on the CNN dataset using sequence-to-sequence generation
- **Parameters:** 4 epochs, batch size 4, AdamW optimizer (lr=2e-5), mixed precision (AMP)
- **Output:** New sentences that rephrase and synthesize the original content
### 2. Extractive Summarization — KNN + TF-IDF
 
- **Technique:** Sentences scored using word frequency and normalized TF-IDF scores
- **Similarity:** Cosine similarity computed between TF-IDF vectors
- **Selection:** Top-N highest-scoring sentences aggregated as the summary
- **No training required** — operates directly on input text at inference time
---
 
## 📊 Model Performance
 
### Abstractive (BART-large) — ROUGE Scores across 4 Epochs
 
| Epoch | Train Loss | Val Loss | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|-----------|----------|---------|---------|---------|
| 1 | 1.285 | 0.973 | 0.469 | 0.227 | 0.309 |
| 2 | 0.927 | 0.968 | 0.469 | 0.228 | 0.310 |
| 3 | 0.827 | 0.978 | 0.470 | 0.228 | 0.311 |
| 4 | 0.740 | 1.000 | 0.474 | 0.231 | 0.312 |
 
> Test Set Results: **ROUGE-1: 0.473 | ROUGE-2: 0.230 | ROUGE-L: 0.310**
 
### Extractive (KNN + TF-IDF)
 
- **Cosine Similarity Score:** `0.6838` (between full article and generated summary)
---
 
## 🚀 Installation
 
### Prerequisites
 
- Python 3.9+
- pip
- GPU recommended for BART inference (CUDA-compatible)
### 1. Clone the Repository
 
```bash
git clone https://github.com/<your-username>/newsnutshell.git
cd newsnutshell
```
 
### 2. Create a Virtual Environment
 
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```
 
### 3. Install Dependencies
 
```bash
pip install -r requirements.txt
```
 
### 4. Download NLTK Data
 
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```
 
### 5. Set Up Environment Variables
 
Create a `.env` file in the root directory:
 
```
NEWS_API_KEY=your_news_api_key_here
SECRET_KEY=your_flask_secret_key_here
```
 
Get your free News API key at [newsapi.org](https://newsapi.org/).
 
### 6. Add the BART Model
 
Place your fine-tuned BART model and tokenizer in:
```
models/best_bartlarge_model/
models/best_bartlarge_tok/
```
 
Or the app will fall back to loading `facebook/bart-large-cnn` directly from Hugging Face.
 
### 7. Run the Application
 
```bash
python app.py
```
 
Visit `http://127.0.0.1:5000` in your browser.
 
---
 
## 💻 Usage
 
1. **Register** a new account or **Login** as an existing user.
2. Navigate to the **Summarize** page.
3. Paste or type your news article text.
4. Choose your settings:
   - Summarization Type: `Abstractive` or `Extractive`
   - Number of Keywords (1–5)
   - Output Format: `Paragraph` or `Bullet Points`
   - Number of Sentences (for Extractive, 1–10)
5. Click **Generate Summary**.
6. View the summary, extracted keywords, and cosine similarity score.
7. Check **History** to revisit past summaries.
8. Submit **Feedback** to help improve the system.
---
 
## 📁 Repository Structure
 
```
newsnutshell/
│
├── app.py                        # Flask application entry point
│
├── summarizers/
│   ├── abstractive.py            # BART-large abstractive summarizer
│   └── extractive.py             # KNN + TF-IDF extractive summarizer
│
├── static/
│   └── css/                      # Stylesheets
│
├── templates/
│   ├── login.html
│   ├── register.html
│   └── index.html
│
├── notebooks/
│   ├── abstractive_bart.ipynb    # BART training notebook (Kaggle/Colab)
│   └── extractive_knn.ipynb      # KNN extractive summarization notebook
│
└── README.md
```
 
---
 
## ⚠️ Challenges & Limitations
 
- **Hardware Constraints:** Full fine-tuning of BART-large required significant GPU memory; training was limited to 4 epochs.
- **Data Quality:** The CNN/DailyMail dataset contained truncated articles and noisy metadata which affected summary coherence.
- **Overfitting:** Validation loss slightly increased by Epoch 4 (0.968 → 1.000), indicating the model may over-optimize on training data.
- **KNN Bias:** TF-IDF-based extraction is biased toward word frequency over semantic depth, potentially missing contextually important but infrequent sentences.
- **Single-Source Bias:** The CNN dataset reflects one outlet's writing style, which may not generalize to all news sources.
---
 
## 🔮 Future Work
 
- 🔁 **Reinforcement Learning** — Optimize summaries based on user feedback and custom rewards (coherence, conciseness)
- 🌐 **Multilingual Support** — Extend to non-English sources using translation models (MarianMT)
- 🤖 **BERT-based Extractive Scoring** — Replace TF-IDF with contextual sentence embeddings for deeper semantic understanding
- 🔀 **Hybrid Summarization** — Use KNN to seed key sentences, then rephrase using BART for balanced precision and readability
- 📱 **Browser Extension** — One-click summarization for any webpage
- 📊 **Predictive Analytics Dashboard** — Trend graphs and topic heat maps from user queries
- 😊 **Sentiment Tagging** — Label summaries with tone (positive/negative/neutral)
---
 
## 📚 References
 
- **Dataset:** [CNN/DailyMail on Hugging Face](https://huggingface.co/datasets/cnn_dailymail)
- **Model:** [facebook/bart-large-cnn on Hugging Face](https://huggingface.co/facebook/bart-large-cnn)
- **News API:** [https://newsapi.org](https://newsapi.org)
- **ROUGE Metric:** [rouge-score library](https://pypi.org/project/rouge-score/)
- **NLTK:** [https://www.nltk.org](https://www.nltk.org)
- **scikit-learn:** [https://scikit-learn.org](https://scikit-learn.org)
- Development supported by Google Colab, Kaggle, and VS Code
---
 


# TrendTales AI 🎧

> Real-Time Trends → NLP Analysis → Fine-Tuned Multi-Label BERT → RAG → Story Generation → TTS

An NLP graduation project that collects real news trends, classifies them into multi-label categories using a fine-tuned BERT model, and generates personalized audio stories.

---

## 🚀 Pipeline Overview

```
1. Data Collection     → google_news_collector.py / rss_news.py / Newsapi_data_collection.py
2. Preprocessing       → merge_news.py      (clean & deduplicate articles)
3. Trend Ranking       → trend_ranking_v2.py (score by frequency × recency × source diversity)
4. BERT Classification → bert_classifier/classify.py (multi-label BERT → generates classified_data.json)
5. RAG System          → (RAG Indexer & Vector Store)
6. Story Generation    → (LLM Podcast Script Generator)
7. TTS                 → (Text-To-Speech Audio Generator)
```

---

## 📥 Fine-Tuned Model Weights (Download Link)

The fine-tuned BERT model weights (~438 MB) are gitignored to keep the repository lightweight. You can download the pre-trained model directly without retraining:

👉 **[Download Fine-Tuned BERT Model Weights (Google Drive)](https://drive.google.com/file/d/1YNesMp61Eww6z1W_bxNPU-3ke9BsQPU-/view?usp=sharing)**

**Installation Instructions:**
1. Download `saved_model.zip` from the Drive link above.
2. Extract the contents directly into the `bert_classifier/saved_model/` directory.

---

## ⚙️ Setup & Installation

**Prerequisites:** Python 3.10+, pip, NVIDIA GPU recommended

```bash
git clone https://github.com/tonynagyy/TrendTales.git
cd TrendTales

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
cp .env.example .env           # Add your API keys inside .env
```

---

## 📦 Generating & Classifying the Data

> ⚠️ Data files are **not included** in this repo (they are gitignored). Run the pipeline in order to generate and classify them:

```bash
# Step 1: Merge & Deduplicate collected articles
python "Data preprocessing/mrege_news.py"

# Step 2: Rank Top News Trends
python "Data preprocessing/trend_ranking_v2.py"

# Step 3: Run Multi-Label BERT Classifier
python bert_classifier/classify.py
```

### 📄 Final Preprocessed Output File
The final output file generated for the RAG system is **`Data preprocessing/Data/classified_data.json`**. Each article contains:
- `categories`: Multi-label category array (e.g. `["Business", "Technology"]`)
- `confidence`: Confidence score map for each category (e.g. `{"Technology": 0.9758, "Business": 0.8762}`)
- `category`: Primary category string for backward compatibility

---

## 🤖 Multi-Label BERT Classifier (Feature 2)

Fine-tuned `bert-base-uncased` on Kaggle Multi-Label News Data across **8 categories**:  
`Politics | Economy | Business | Technology | Climate | Science | Education | International Affairs`

```bash
# Train the model (optional if you downloaded weights from Drive)
python bert_classifier/train.py

# Evaluate performance & generate per-category metrics chart
python bert_classifier/evaluate.py

# Classify pipeline articles -> generates classified_data.json
python bert_classifier/classify.py
```

### 📊 Evaluation Metrics:
- **Macro F1 Score:** **85.86%**
- **Micro F1 Score:** **84.43%**
- **Weighted F1 Score:** **84.44%**
- **Sample Output:** Classified **7,120 articles** into multi-label tags with high confidence.

---

## 📁 Project Structure

```
TrendTales/
├── Data preprocessing/
│   ├── google_news_collector.py
│   ├── rss_news.py
│   ├── Newsapi_data_collection.py
│   ├── mrege_news.py
│   ├── trend_ranking_v2.py
│   └── Data/               ← gitignored, contains classified_data.json
├── bert_classifier/
│   ├── config.py
│   ├── train.py
│   ├── evaluate.py
│   ├── classify.py
│   ├── data/               ← gitignored (Kaggle dataset CSV)
│   ├── saved_model/        ← gitignored (Download from Drive link)
│   └── results/            ← evaluation reports & charts
├── .env.example
├── requirements.txt
└── README.md
```
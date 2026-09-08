# TrendTales AI 🎧

> Real-Time Trends → NLP Analysis → Fine-Tuned BERT → RAG → Story Generation → TTS

An NLP graduation project that collects real news trends, classifies them using a fine-tuned BERT model, and generates personalized audio stories.

---

## 🚀 Pipeline Overview

```
1. Data Collection     → data_collection.py       (NewsAPI + Reddit)
2. Preprocessing       → preprocessing.py          (clean & deduplicate)
3. Trend Ranking       → trend_ranking.py           (score by frequency × recency × source diversity)
4. BERT Classification → bert_classifier/train.py  (fine-tune on AG News → classify real articles)
5. RAG System          → (coming soon)
6. Story Generation    → (coming soon)
7. TTS                 → (coming soon)
```

---

## ⚙️ Setup

**Prerequisites:** Python 3.10+, pip, NVIDIA GPU recommended

```bash
git clone https://github.com/your-username/TrendTales.git
cd TrendTales

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
cp .env.example .env           # Add your API keys inside .env
```

---

## 📦 Generating the Data

> ⚠️ Data files are **not included** in this repo (they are gitignored). Run the pipeline in order to generate them:

```bash
# Step 1: Collect news articles
python "Data preprocessing/data_collection.py"

# Step 2: Clean & deduplicate
python "Data preprocessing/preprocessing.py"

# Step 3: Rank trends
python "Data preprocessing/trend_ranking.py"
```

Output files will be created under `Data preprocessing/Data/`:
- `sample_data.json` — raw collected articles
- `cleaned_data.json` — preprocessed articles
- `top_trends.json` — top ranked trends

---

## 🤖 BERT Classifier (Feature 2)

Fine-tunes `bert-base-uncased` on AG News to classify articles into:
`World | Sports | Business | Sci/Tech`

```bash
# Train the model (requires GPU for speed)
python bert_classifier/train.py

# Evaluate (generates confusion matrix + F1 report)
python bert_classifier/evaluate.py

# Classify your collected articles
python bert_classifier/classify.py
```

**Results (20K training samples, 3 epochs):**
- Accuracy: **92.35%**
- Macro F1: **92.45%**
- Training time: ~10 min (RTX 4050)

Output saved to `Data preprocessing/Data/classified_data.json`

---

## 🔑 Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```
NEWS_API_KEY=your_newsapi_key
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
```

---

## 📁 Project Structure

```
TrendTales/
├── Data preprocessing/
│   ├── data_collection.py
│   ├── preprocessing.py
│   ├── trend_ranking.py
│   └── Data/               ← gitignored, generated locally
├── bert_classifier/
│   ├── config.py
│   ├── train.py
│   ├── evaluate.py
│   ├── classify.py
│   ├── saved_model/        ← gitignored, generated after training
│   └── results/            ← gitignored, generated after evaluate
├── .env.example
├── requirements.txt
└── README.md
```
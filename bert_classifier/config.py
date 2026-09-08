# OPTIONS:
#   "bert-base-uncased"    ← our choice: 110M params, fast, well-documented
#   "bert-large-uncased"   → 340M params, ~2% more accurate, but 3x slower
#   "distilbert-base-uncased" → 66M params, 40% faster but ~3% less accurate
#   "roberta-base"         → similar to BERT but slightly better on some tasks

MODEL_NAME = "bert-base-uncased"

# OPTIONS:
#   "ag_news"           ← our choice: 4 categories, 120K samples, fast to load
#   "dbpedia_14"        → 14 categories, 560K samples — overkill for our demo
#   "yahoo_answers_topics" → 10 categories, messy text, slower
#   Custom dataset      → needs manual labeling, not feasible in 3 days

DATASET_NAME = "fancyzhx/ag_news"
NUM_LABELS = 4
LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}


TRAIN_SUBSET_SIZE = 20_000
TEST_SUBSET_SIZE  = 2_000   


MAX_LENGTH = 128



EPOCHS = 3


BATCH_SIZE = 16

LEARNING_RATE = 2e-5

# WEIGHT DECAY:
#   WHY 0.01? A small regularization penalty to prevent overfitting.
#             Standard for BERT. Don't change this.
WEIGHT_DECAY = 0.01


MODEL_SAVE_PATH   = "bert_classifier/saved_model"
RESULTS_SAVE_PATH = "bert_classifier/results"
DATA_PATH         = "Data preprocessing/Data/cleaned_data.json"

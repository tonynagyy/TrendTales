# ==========================================
# BERT MULTI-LABEL CLASSIFIER CONFIGURATION
# ==========================================

MODEL_NAME = "bert-base-uncased"

# The 8 actual Multi-Label Categories present in Kaggle "News Article Classification Data"
LABEL_MAP = {
    0: "Politics",
    1: "Economy",
    2: "Business",
    3: "Technology",
    4: "Climate",
    5: "Science",
    6: "Education",
    7: "International Affairs"
}

LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}
NUM_LABELS = len(LABEL_MAP)

# Threshold for assigning a category in multi-label output
MULTI_LABEL_THRESHOLD = 0.50

# Dataset limits for training
TRAIN_SUBSET_SIZE = 25_000
TEST_SUBSET_SIZE  = 3_000   

MAX_LENGTH = 128
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# Folder Paths
DATA_DIR          = "bert_classifier/data"
MODEL_SAVE_PATH   = "bert_classifier/saved_model"
RESULTS_SAVE_PATH = "bert_classifier/results"
INPUT_DATA_PATH   = "Data preprocessing/Data/cleaned_data.json"
OUTPUT_DATA_PATH  = "Data preprocessing/Data/classified_data.json"

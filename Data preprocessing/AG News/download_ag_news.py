from datasets import load_dataset
from collections import Counter
import os

dataset = load_dataset("sh0416/ag_news")

print(dataset)

print("Train:")
print(dataset["train"][0])

print("Test:")
print(dataset["test"][0])

train_labels = Counter(dataset["train"]["label"])
test_labels = Counter(dataset["test"]["label"])

print("Train label distribution:")
print(train_labels)

print("Test label distribution:")
print(test_labels)

os.makedirs("Data preprocessing/Ag News/raw", exist_ok=True)

dataset["train"].to_json("Data preprocessing/Ag News/raw/train.jsonl")
dataset["test"].to_json("Data preprocessing/Ag News/raw/test.jsonl")

print("Raw AG News data saved successfully.")
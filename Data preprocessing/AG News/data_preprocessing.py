import pandas as pd
import os

os.makedirs("Data preprocessing/Ag News/processed", exist_ok=True)

train = pd.read_json("Data preprocessing/Ag News/raw/train.jsonl", lines=True)
test = pd.read_json("Data preprocessing/Ag News/raw/test.jsonl", lines=True)

print("Train shape:", train.shape)
print("Test shape:", test.shape)

print("\nTrain missing values:")
print(train.isnull().sum())

print("\nTest missing values:")
print(test.isnull().sum())

print("\nTrain duplicates:", train.duplicated().sum())
print("Test duplicates:", test.duplicated().sum())

print("\nTrain labels:")
print(sorted(train["label"].unique()))

print("\nTest labels:")
print(sorted(test["label"].unique()))

train["text"] = train["title"].fillna("") + " " + train["description"].fillna("")
test["text"] = test["title"].fillna("") + " " + test["description"].fillna("")

train["text"] = train["text"].str.strip()
test["text"] = test["text"].str.strip()

train = train[["text", "label"]]
test = test[["text", "label"]]

train.to_json("Data preprocessing/Ag News/processed/train.jsonl", orient="records", lines=True)
test.to_json("Data preprocessing/Ag News/processed/test.jsonl", orient="records", lines=True)

final_train = pd.read_json("Data preprocessing/Ag News/processed/train.jsonl", lines=True)
final_test = pd.read_json("Data preprocessing/Ag News/processed/test.jsonl", lines=True)

print("\nFinal Train:")
print(final_train.shape)
print(final_train.columns.tolist())

print("\nFinal Test:")
print(final_test.shape)
print(final_test.columns.tolist())

print("\nFinal Train missing values:")
print(final_train.isnull().sum())

print("\nFinal Test missing values:")
print(final_test.isnull().sum())
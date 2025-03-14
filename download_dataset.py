from datasets import load_dataset
import pandas as pd

# Load AG News dataset from Hugging Face
dataset = load_dataset("ag_news")

# Convert train and test datasets to pandas DataFrames
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

# Save to CSV files
train_df.to_csv("ag_news_train.csv", index=False)
test_df.to_csv("ag_news_test.csv", index=False)

print("✅ Dataset saved as CSV: ag_news_train.csv & ag_news_test.csv")

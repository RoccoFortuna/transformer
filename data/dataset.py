import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(self, split="train", tokenizer_name="bert-base-uncased", max_length=50):
        self.dataset = load_dataset("ag_news", split=split)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]
        label = self.dataset[idx]["label"]

        encoded_text = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        input_ids = encoded_text["input_ids"].squeeze(0)

        return input_ids, torch.tensor(label)


def get_dataloaders(batch_size=32, max_length=50):
    train_dataset = TextDataset(split="train", max_length=max_length)
    test_dataset = TextDataset(split="test", max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
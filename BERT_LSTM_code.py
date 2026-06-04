import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

messages = pd.read_csv("/content/dataset_labels.csv")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class DatasetClass(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.values
        self.labels = labels.values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(self.texts[idx], padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx])
        }

X_train, X_test, y_train, y_test = train_test_split(
    messages["text"], messages["label"], test_size=0.3, stratify=messages["label"]
)

train_loader = DataLoader(DatasetClass(X_train, y_train), batch_size=4, shuffle=True)
test_loader = DataLoader(DatasetClass(X_test, y_test), batch_size=4)

class Bert_GRU_MC(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.gru = nn.GRU(
            input_size=768,
            hidden_size=128,
            batch_first=True,
            bidirectional=False
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 2)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        _, h = self.gru(x)

        h = h[-1]

        h = self.dropout(h)

        return self.fc(h)

    def enable_dropout(self):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Bert_GRU_MC().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):
    model.train()
    for batch in train_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        out = model(ids, mask)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

model.eval()
model.enable_dropout()

y_pred, y_true = [], []

with torch.no_grad():
    for batch in test_loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["label"].to(device)

        mc = []
        for _ in range(10):
            probs = torch.softmax(model(ids, mask), dim=1)
            mc.append(probs.unsqueeze(0))

        mc = torch.cat(mc, dim=0)
        mean = mc.mean(dim=0)

        pred = torch.argmax(mean, dim=1)

        y_pred.extend(pred.cpu().numpy())
        y_true.extend(y.cpu().numpy())

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))
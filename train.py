import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
import os

# Đọc dữ liệu IMDB
df = pd.read_csv("IMDB Dataset.csv")

# Hàm tiền xử lý văn bản
def process(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = ' '.join(text.split())
    return text

# Class Dataset
class MyData(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Khởi tạo tokenizer và mô hình
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tiền xử lý dữ liệu
df['text'] = df['review'].apply(process)
df['label'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Chia dữ liệu
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

# Tạo dataset và dataloader
train_dataset = MyData(train_texts.to_list(), train_labels.to_list(), tokenizer)
test_dataset = MyData(test_texts.to_list(), test_labels.to_list(), tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Thiết bị
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Đóng băng các layer của BERT, chỉ huấn luyện classifier
for name, param in model.named_parameters():
    if 'classifier' not in name:  # Đóng băng tất cả trừ classifier
        param.requires_grad = False

# Nếu muốn huấn luyện thêm layer Encoder cuối (ví dụ: layer 11), bỏ comment đoạn sau
# for name, param in model.named_parameters():
#     if 'classifier' in name or 'bert.encoder.layer.11' in name:  # Huấn luyện classifier và layer 11
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

# In ra các tham số được huấn luyện
print("Các tham số được huấn luyện:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

# Thiết lập optimizer (chỉ tối ưu các tham số không bị đóng băng)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

# Hàm đánh giá
def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    accuracy = correct / total
    return accuracy

# Huấn luyện và lưu mô hình tốt nhất
best_accuracy = 0.0
best_model_path = 'best_bert_model.pth'

model.train()
for epoch in range(10):  # Huấn luyện 3 epoch
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Tính loss trung bình
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    # Đánh giá trên tập test
    accuracy = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1}, Test Accuracy: {accuracy:.4f}")

    # Lưu mô hình nếu tốt hơn
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with accuracy: {best_accuracy:.4f}")

# Tải mô hình tốt nhất để kiểm tra
model.load_state_dict(torch.load(best_model_path))
model.eval()
final_accuracy = evaluate(model, test_loader, device)
print(f"Final Accuracy with best model: {final_accuracy:.4f}")

# Hàm dự đoán trên câu mới
def predict_sentiment(text):
    model.eval()
    text = process(text)  # Tiền xử lý
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=256,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return "Tích cực" if prediction == 1 else "Tiêu cực"

# Thử dự đoán
new_text = "This movie is fantastic!"
print(f"Câu: {new_text} → Sentiment: {predict_sentiment(new_text)}")
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from bs4 import BeautifulSoup
import re

# Hàm tiền xử lý văn bản (giống với huấn luyện)
def process(text):
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
    text = ' '.join(text.split())
    return text

# Hàm dự đoán cảm xúc
def predict_sentiment(model, tokenizer, text, device, max_len=256):
    model.eval()
    text = process(text)  # Tiền xử lý văn bản
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
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

# Main
def main():
    # Thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Khởi tạo tokenizer và mô hình
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Đường dẫn đến mô hình đã lưu
    model_path = 'best_bert_model.pth'

    # Load mô hình
    try:
        model.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file {model_path} not found. Please train the model first.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Chuyển mô hình sang thiết bị
    model.to(device)

    # Danh sách các câu để dự đoán
    test_sentences = [
        "This movie is fantastic!",
        "I hated this movie, it was boring.",
        "The plot was amazing and the actors were great.",
        "What a terrible experience, never watching again.",
        "I love movie, that's good",
        "The actor is so ugly",
        "This movie is terrible",
        "That's fantastic"
    ]

    # Dự đoán cho từng câu
    print("\nDự đoán cảm xúc:")
    for sentence in test_sentences:
        sentiment = predict_sentiment(model, tokenizer, sentence, device)
        print(f"Câu: {sentence} → Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
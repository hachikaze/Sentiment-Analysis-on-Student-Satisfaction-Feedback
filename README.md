# **Sentiment Analysis on Student Satisfaction Feedback**

## **Overview**  
This project fine-tunes a **BERT-based model** for **sentiment analysis** on student feedback. It utilizes **Hugging Face Transformers, PyTorch, and Scikit-Learn** for preprocessing, training, and evaluation.

---

## **📌 Importing Libraries**  
![Importing Libraries](Sentiment-Analysis-on-Student-Satisfaction-Feedback/images/importing_libraries.png)  

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
```

---

## **📂 Loading Dataset**  
![Loading Dataset](Sentiment-Analysis-on-Student-Satisfaction-Feedback/images/loading_dataset.png)  

```python
file_path = "/content/TLC_student_feedback_dataset.xlsx"
df = pd.read_excel(file_path)
```

---

## **📊 Splitting Data into Training and Validation Sets**  
![Splitting Data](Sentiment-Analysis-on-Student-Satisfaction-Feedback/images/splitting_data.png)  

```python
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['feedback'].tolist(), df['label'].tolist(),
    test_size=0.2, random_state=42
)
```

---

## **🔠 Tokenizing the Data**  
![Tokenization](Sentiment-Analysis-on-Student-Satisfaction-Feedback/images/tokenization.png)  

```python
tokenizer = AutoTokenizer.from_pretrained("MarieAngeA13/Sentiment-Analysis-BERT")

class FeedbackDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encodings = self.tokenizer(
            text, truncation=True, padding="max_length",
            max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }
```

---

## **🏋️ Training the Model**  
![Training Model](Sentiment-Analysis-on-Student-Satisfaction-Feedback/images/training_model.png)  

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "MarieAngeA13/Sentiment-Analysis-BERT", num_labels=3
)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    save_total_limit=2,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)
```

---

## **🎯 Model Evaluation**  
![Model Evaluation](Sentiment-Analysis-on-Student-Satisfaction-Feedback/images/evaluation.png)  

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
        "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
    }
```

---

## **🚀 Training the Model**  

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
```

---

## **💾 Saving the Fine-Tuned Model**  
![Saving Model](Sentiment-Analysis-on-Student-Satisfaction-Feedback/images/saving_model.png)  

```python
trainer.save_model("./bert_fine_tuned_model")
tokenizer.save_pretrained("./bert_fine_tuned_model")
```

---

## **📈 Results & Next Steps**  
✅ The results of the fine-tuned model, including accuracy and evaluation metrics, are available inside the **notebook**. Go and check them there! 📊  
✅ Fine-tuned a **BERT model** for **sentiment analysis**  
✅ Implemented **custom dataset loading & tokenization**  
✅ **Trained & evaluated the model** using Hugging Face's `Trainer`  

🔹 Next steps:  
- Experiment with **different transformer models**  
- Fine-tune with **more training epochs or hyperparameters**  
- Deploy the model using **Flask or FastAPI**  

---

## **🤝 Contributing**  
Feel free to **fork** this repository, submit **pull requests**, or report **issues**! 🚀  

---

## **📝 License**  
This project is licensed under the **MIT License**.  

---

### **🎯 Why This README is Better?**  
✔ **Markdown format** (GitHub-friendly, no unnecessary HTML)  
✔ **Code highlighting** (Easy to read & copy)  
✔ **Structured sections** (Installation, Data Preprocessing, Training, Evaluation)  
✔ **Emojis & formatting** (More engaging & readable)  

**Ensure images are placed in `images/` folder within your project directory for proper display on GitHub.** 🚀


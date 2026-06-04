# E-Bert
# E-BERT: Fake News Detection Using Deep Learning

## 📌 Project Overview

E-BERT is a Fake News Detection system developed using advanced Deep Learning techniques. The project utilizes BERT embeddings combined with LSTM and GRU models to classify news articles as **Real** or **Fake**.

The goal of this project is to combat misinformation by automatically detecting fake news from textual content with high accuracy.

---

## 🚀 Features

* Fake News Classification
* BERT-based Text Embedding
* LSTM Model Implementation
* GRU Model Implementation
* Data Preprocessing and Cleaning
* Performance Evaluation
* Binary Classification (Real/Fake)

---

## 🛠️ Technologies Used

* Python
* BERT (Bidirectional Encoder Representations from Transformers)
* TensorFlow / Keras
* LSTM
* GRU
* Pandas
* NumPy
* Scikit-learn

---

## 📂 Project Structure

```text
E-Bert/
│
├── BERT_GRU_code.py
├── BERT_LSTM_code.py
├── modifieddataset.csv
├── Final_Document.pdf
├── Fake news detecction base paper.pdf
└── README.md
```

---

## 📊 Dataset

The dataset contains news articles labeled as:

* **0 → Real News**
* **1 → Fake News**

### Dataset Columns

| Column | Description              |
| ------ | ------------------------ |
| text   | News article content     |
| label  | Real/Fake classification |

Dataset File:

```text
modifieddataset.csv
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/DineshKiran11/E-Bert.git
```

### Navigate to the Project Directory

```bash
cd E-Bert
```

### Install Dependencies

```bash
pip install pandas numpy scikit-learn tensorflow transformers
```

---

## ▶️ Running the Project

### Run BERT + GRU Model

```bash
python BERT_GRU_code.py
```

### Run BERT + LSTM Model

```bash
python BERT_LSTM_code.py
```

---

## 🔍 Methodology

1. Data Collection
2. Data Preprocessing
3. Text Cleaning
4. BERT Tokenization
5. Feature Extraction
6. Model Training
7. Model Evaluation
8. Fake News Prediction

---

## 📈 Results

The developed models effectively classify news articles by leveraging:

* BERT for contextual text understanding
* LSTM for sequential pattern learning
* GRU for efficient sequence modeling

The system demonstrates strong performance in identifying fake news content from real news articles.

---

## 🎯 Future Enhancements

* Real-Time Fake News Detection
* Web Application Deployment
* Mobile Application Integration
* Multilingual Fake News Detection
* Enhanced Model Accuracy

---

## 👨‍💻 Author

**Dinesh Kiran**

GitHub:
https://github.com/DineshKiran11

LinkedIn:
https://www.linkedin.com/in/kambampati-dinesh-kiran-427aab304

---

## 📄 License

This project is intended for educational, research, and learning purposes.

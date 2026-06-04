# E-Bert
# Detection and Classification of Fake news using Deep learning technique E-Bert

## 📌 Project Overview

E-BERT is a Fake News Detection system developed using advanced Deep Learning techniques. The project utilizes BERT embeddings combined with LSTM and GRU models to classify news articles as **Real** or **Fake**.

To improve prediction reliability, the proposed framework incorporates **Monte Carlo Dropout** for uncertainty estimation. This enables the model not only to classify news articles but also to estimate the confidence of its predictions.

The goal of this project is to combat misinformation by automatically detecting fake news from textual content with high accuracy and reliability.

---

## 🚀 Features

* Fake News Classification
* BERT-based Text Embedding
* LSTM Model Implementation
* GRU Model Implementation
* Monte Carlo Dropout for Uncertainty Estimation
* Confidence-Aware Predictions
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
* Monte Carlo Dropout
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

---

## 📂 Project Structure

```text
E-Bert/
│
├── BERT_GRU_code.py
├── BERT_LSTM_code.py
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

### Processed Dataset

The processed dataset used for model training and testing can be downloaded from:

https://drive.google.com/file/d/1Xz62VO4HotE-jlHUyyRs10PJ5-2PYfFE/view?usp=sharing

**File Name:**

```text
modifieddataset.csv
```

### Real-World Dataset

The original real-world dataset used for data collection and preprocessing is available at:

https://drive.google.com/drive/folders/1dGdoswSZPqlU_tzQUGgk1GwC_Ocfl4qA?usp=drive_link

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
pip install pandas numpy scikit-learn tensorflow transformers matplotlib
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
7. Monte Carlo Dropout-Based Uncertainty Estimation
8. Model Evaluation
9. Fake News Prediction

---

## 📈 Results

The developed models effectively classify news articles by leveraging:

* BERT for contextual text understanding
* LSTM for sequential pattern learning
* GRU for efficient sequence modeling
* Monte Carlo Dropout for confidence estimation

The system demonstrates strong performance in identifying fake news content from real news articles while providing prediction confidence for improved reliability.

---

## 🎯 Future Enhancements

* Real-Time Fake News Detection
* Dynamic Uncertainty Thresholds
* Human-in-the-Loop Verification
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

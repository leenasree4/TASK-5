# Task 5: Consumer Complaint Text Classification

**Author:** Leena Sree S
**Date:** 28.09.2025 

## Overview

This project performs **text classification** on the [Consumer Complaint Dataset](https://catalog.data.gov/dataset/consumer-complaint-database) to classify complaints into four categories:

| Label | Category |
|-------|----------|
| 0     | Credit reporting, repair, or other |
| 1     | Debt collection |
| 2     | Consumer Loan |
| 3     | Mortgage |

The project includes the following steps:

1. Explanatory Data Analysis and Feature Engineering  
2. Text Preprocessing  
3. Training Multiple Classification Models (Logistic Regression, Naive Bayes, SVM)  
4. Comparing Model Performance  
5. Model Evaluation  
6. Making Predictions  

---

## 1. Explanatory Data Analysis (EDA)

- **Dataset Size:** [Insert dataset size]  
- **Number of Classes:** 4  
- **Distribution of Classes:**  

![Class Distribution](screenshots/class_distribution.png)


- Observations:  
  - [Write a few lines about class imbalance, if any]  
  - [Other interesting observations]

---

## 2. Feature Engineering & Text Preprocessing

- Removed null values  
- Cleaned text (lowercasing, removing punctuation, numbers, links)  
- Converted text into **TF-IDF vectors** (max 5000 features)  

![Text Preprocessing Screenshot](screenshots/text_preprocessing.png)  


---

## 3. Model Training

### Logistic Regression

- Trained using `LogisticRegression(max_iter=200)`  
- Accuracy: `[Insert Accuracy]`  

![Logistic Regression Screenshot](screenshots/log_reg.png)  


### Naive Bayes

- Trained using `MultinomialNB()`  
- Accuracy: `[Insert Accuracy]`  

![Naive Bayes Screenshot](screenshots/naive_bayes.png)  


### Support Vector Machine (SVM)

- Trained using `LinearSVC()`  
- Accuracy: `[Insert Accuracy]`  

![SVM Screenshot](screenshots/svm.png)  


---

## 4. Model Comparison

| Model               | Accuracy |
|--------------------|----------|
| Logistic Regression | `[Insert]` |
| Naive Bayes         | `[Insert]` |
| SVM                 | `[Insert]` |

![Model Comparison Screenshot](screenshots/model_comparison.png)  


Observations:  
- [Write a few lines about which model performed best and why]  

---

## 5. Example Prediction

- Sample complaint text: `"I am not happy with my mortgage lender, they keep charging fees!"`  
- Predicted category: `[Insert prediction]`  

![Sample Prediction Screenshot](screenshots/sample_prediction.png)  


---

## 6. Instructions to Run

1. Clone the repository:  
   ```bash
   git clone [Your Repository URL]

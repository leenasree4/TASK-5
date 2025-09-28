# Consumer Complaint Text Classification
# Categories:
# 0 - Credit reporting, repair, or other
# 1 - Debt collection
# 2 - Consumer Loan
# 3 - Mortgage

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# -------------------------------
# 1. Load dataset
# -------------------------------
print("Loading dataset...")
url = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
df = pd.read_csv(url, compression='zip')

# Select only relevant columns
df = df[['Product', 'Consumer complaint narrative']].dropna()

# -------------------------------
# 2. Map Products -> Categories
# -------------------------------
def map_product(product):
    if "Credit reporting" in product or "Credit card or prepaid card" in product:
        return 0
    elif "Debt collection" in product:
        return 1
    elif "Consumer Loan" in product or "Vehicle loan" in product:
        return 2
    elif "Mortgage" in product:
        return 3
    else:
        return 0  # Map everything else to 0

df['label'] = df['Product'].apply(map_product)

print("Dataset size after mapping:", df.shape)

# -------------------------------
# 3. Text Preprocessing
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

df['clean_text'] = df['Consumer complaint narrative'].apply(clean_text)

# -------------------------------
# 4. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# -------------------------------
# 5. TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# -------------------------------
# 6. Train Models Separately
# -------------------------------

# Logistic Regression
print("\nTraining Logistic Regression...")
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train_tfidf, y_train)
y_pred = log_reg.predict(X_test_tfidf)
log_reg_acc = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", log_reg_acc)
print(classification_report(y_test, y_pred))

# Naive Bayes
print("\nTraining Naive Bayes...")
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred = nb.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, y_pred)
print("Naive Bayes Accuracy:", nb_acc)
print(classification_report(y_test, y_pred))

# Support Vector Machine
print("\nTraining SVM...")
svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)
y_pred = svm.predict(X_test_tfidf)
svm_acc = accuracy_score(y_test, y_pred)
print("SVM Accuracy:", svm_acc)
print(classification_report(y_test, y_pred))

# -------------------------------
# 7. Final Model Comparison
# -------------------------------
results_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Naive Bayes", "SVM"],
    "Accuracy": [log_reg_acc, nb_acc, svm_acc]
})

print("\nFinal Model Comparison:")
print(results_df)

# Plot bar chart of accuracies
sns.barplot(x="Model", y="Accuracy", data=results_df)
plt.title("Model Comparison")
plt.show()

# -------------------------------
# 8. Example Prediction
# -------------------------------
sample_text = ["I am not happy with my mortgage lender, they keep charging fees!"]
sample_tfidf = vectorizer.transform(sample_text)
prediction = svm.predict(sample_tfidf)
inv_categories = {0:"Credit reporting, repair, or other", 1:"Debt collection", 2:"Consumer Loan", 3:"Mortgage"}
print("\nSample Prediction:", inv_categories[prediction[0]])

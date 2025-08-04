# 📧 Email Spam Detection

A Machine Learning project to classify emails as *Spam* or *Ham (Not Spam)* using the *Multinomial Naive Bayes* algorithm. The project includes complete data preprocessing, feature extraction using Bag-of-Words, model training, evaluation, and accuracy visualization.

---

## 🔍 Problem Statement

Spam emails waste time, can be dangerous, and are a major annoyance. This project aims to build a smart email classifier that automatically detects whether an email is spam or not using machine learning.

---

## 📁 Dataset

- The dataset contains labeled emails as spam or ham.
- Preprocessing includes removing stop words, punctuation, and converting text to lowercase.

---

## 🛠 Tools & Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

---

## 📊 Machine Learning Algorithm

- *Model used:* Multinomial Naive Bayes
- *Vectorization:* CountVectorizer (Bag-of-Words Model)
- *Accuracy achieved:* *98.92%*

---

## 🧠 Steps Involved

1. *Data Cleaning & Preprocessing*
   - Remove punctuation, lowercase conversion, remove stop words.
2. *Feature Extraction*
   - Convert text to numerical vectors using CountVectorizer.
3. *Model Building*
   - Train Multinomial Naive Bayes on training data.
4. *Evaluation*
   - Accuracy score and confusion matrix.
5. *Prediction*
   - Classify new email input as Spam or Ham.

---

## 📈 Output Example

```python
Input: "Congratulations! You've won a free prize. Click here to claim."
Prediction: Spam ✅


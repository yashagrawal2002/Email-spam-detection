import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load data
df = pd.read_csv("spam.csv", sep='\t', header=None, names=['label', 'message'], encoding='latin-1')

# 2. Encode labels: 'ham' = 0, 'spam' = 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 4. Vectorization (Bag of Words)
vectorizer = CountVectorizer(stop_words='english')
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

# 5. Train model
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_vectors)

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred))

# 7. Test prediction (optional)
while True:
    user_input = input("\n📨 Enter a message (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)
    print("🔍 Prediction:", "Spam" if prediction[0] == 1 else "Ham")
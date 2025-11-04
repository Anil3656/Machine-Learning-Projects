# 📄 Fake News Detection System
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load Dataset
data = pd.read_csv(r"C:\Users\aneel.kumar\OneDrive - IMCS Group\Desktop\Aneel\Naresh_IT\Fake New Detection Using ML-NLP\train.csv")

# Display basic info
print("✅ Dataset Loaded Successfully!")
print(f"Total samples: {data.shape[0]}")
print(data.head(10))


# Step 2: Handle Missing Data
data = data.dropna(subset=['title', 'text', 'label'])

# Combine title and text for better context
data['content'] = data['title'] + " " + data['text']

print("✅ Missing data handled!")

# Step 3: Split Data
X = data['content']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print("✅ Data split into training and testing sets!")

# Step 4: Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = vectorizer.fit_transform(X_train)
tfidf_test = vectorizer.transform(X_test)

print("✅ Text data vectorized!")

# Step 5: Train Model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

print("✅ Model trained successfully!")


# ==========================
# Step 6: Evaluate Model
# ==========================
y_pred = model.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)

print("\n📊 Model Evaluation Results:")
print(f"✅ Accuracy: {round(score*100, 2)}%")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# ==========================
# Step 7: Save Model & Vectorizer
# ==========================
joblib.dump(model, "fake_news_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n💾 Model and vectorizer saved successfully!")
print("➡️ fake_news_model.pkl")

print("➡️ vectorizer.pkl")



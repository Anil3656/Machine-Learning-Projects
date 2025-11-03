# spam_detector.py

import pandas as pd
import numpy as np
import string
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# Download stopwords once
nltk.download('stopwords')
from nltk.corpus import stopwords

# Step 1: Load Dataset
data = pd.read_csv(r"C:\Users\aneel.kumar\OneDrive - IMCS Group\Desktop\Aneel\Naresh_IT\Spam_Detection_system\SMSSpamCollection", sep='\t', names=['label', 'message'])

#print("Dataset loaded successfully.")
#print(data.head())

# Adjust column names if needed (depends on dataset version)
data = data.rename(columns={'v1': 'label', 'v2': 'message'})

# Keep only the required columns
data = data[['label', 'message']]

# Convert labels to numeric
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})

print("✅ Dataset Loaded Successfully!")
print(data.head())

# Step 2: Text Cleaning Function
def clean_text(text):
    text = text.lower()                           # lowercase
    text = re.sub(r'\d+', '', text)               # remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])  # remove stopwords
    return text

data['clean_message'] = data['message'].apply(clean_text)

print("\n✅ Text cleaned successfully!")
print(data[['message', 'clean_message']].head())

# Step 3: Split Data
X = data['clean_message']
y = data['label_num']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''print("\n✅ Data split into training and testing sets.")
print(X_train.head())
print(X_test.head())
print(y_train.head())
print(y_test.head())'''

# Step 4: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=3000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

'''print("\n✅ Text vectorized using TF-IDF.")
print(vectorizer)
print(f"Number of features: {X_train_vec.shape[1]}")
print(X_test_vec.shape[1])'''

# Step 5: Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

'''print("\n✅ Model trained successfully!")
print(model)'''

# Step 6: Evaluate
y_pred = model.predict(X_test_vec)
print("\n📊 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 7: Save Model
joblib.dump(model, r'C:\Users\aneel.kumar\OneDrive - IMCS Group\Desktop\Aneel\Naresh_IT\Spam_Detection_system\spam_detector.pkl')
joblib.dump(vectorizer, r'C:\Users\aneel.kumar\OneDrive - IMCS Group\Desktop\Aneel\Naresh_IT\Spam_Detection_system\vectorizer.pkl')

print("\n💾 Model and Vectorizer saved successfully!")


# Step 8: Interactive Prediction Feature

def predict_message(msg):
    # Load saved model and vectorizer
    model = joblib.load(r'C:\Users\aneel.kumar\OneDrive - IMCS Group\Desktop\Aneel\Naresh_IT\Spam_Detection_system\spam_detector.pkl')
    vectorizer = joblib.load(r'C:\Users\aneel.kumar\OneDrive - IMCS Group\Desktop\Aneel\Naresh_IT\Spam_Detection_system\vectorizer.pkl')

    # Clean message using same preprocessing
    msg_clean = re.sub(r'\d+', '', msg.lower())
    msg_clean = msg_clean.translate(str.maketrans('', '', string.punctuation))
    msg_clean = ' '.join([word for word in msg_clean.split() if word not in stopwords.words('english')])

    # Transform and predict
    msg_vec = vectorizer.transform([msg_clean])
    prediction = model.predict(msg_vec)[0]

    # Display result
    if prediction == 1:
        print(f"\n⚠️ The message is **SPAM!**")
    else:
        print(f"\n✅ The message is **HAM (Not Spam)**")

# Run interactively
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter a message to check (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("\n👋 Exiting Spam Detector. Goodbye!")
            break
        predict_message(user_input)
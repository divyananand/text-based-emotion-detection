import pandas as pd
import numpy as np
import re
import string

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK data (run once)
nltk.download('stopwords')

# Sample public dataset: https://www.kaggle.com/datasets/datatattle/emotion-dataset
# Load the dataset
df = pd.read_csv("emotion_dataset.csv")  # Replace with your dataset path
print(df.head())

# Check for missing values
df.dropna(inplace=True)

# Preprocessing function
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

df['clean_text'] = df['text'].apply(preprocess_text)

# Split data
X = df['clean_text']
y = df['emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Pipeline with TF-IDF and Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(solver='liblinear', max_iter=1000))
])

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'clf__C': [0.1, 1, 10],
    'tfidf__ngram_range': [(1, 1), (1, 2)]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

# Best model evaluation
print("Best Parameters:", grid.best_params_)
y_pred = grid.predict(X_test)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

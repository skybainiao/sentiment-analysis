# src/preprocess_data.py
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from src.utils import load_data
import joblib

reviews_path = 'C:/Users/45527/PycharmProjects/sentiment-analysis/datalist/reviews.txt'
labels_path = 'C:/Users/45527/PycharmProjects/sentiment-analysis/datalist/labels.txt'

reviews, labels = load_data(reviews_path, labels_path)

X_train, X_temp, y_train, y_temp = train_test_split(reviews, labels, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

vectorizer = CountVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

# 确保 datalist 目录存在
os.makedirs('data', exist_ok=True)

joblib.dump((X_train_vec, X_val_vec, X_test_vec, y_train, y_val, y_test, vectorizer), 'data/preprocessed_data.pkl')
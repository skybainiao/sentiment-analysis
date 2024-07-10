# src/predict.py
import joblib


model = joblib.load('data/sentiment_model.pkl')
_, _, _, _, _, _, vectorizer = joblib.load('data/preprocessed_data.pkl')


new_reviews = [
    "Cjj said he would buy a luxury bag for me as a birthday gift.",
    "I wouldn't recommend this movie to anyone.",
    "A masterpiece, absolutely wonderful from start to finish."
]
# 将新评论转换为词袋表示
new_reviews_vec = vectorizer.transform(new_reviews)

# 使用模型进行预测
new_reviews_pred = model.predict(new_reviews_vec)

# 打印预测结果
for review, prediction in zip(new_reviews, new_reviews_pred):
    print(f"Review: {review}\nPredicted sentiment: {prediction}\n")
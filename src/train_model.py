# src/train_model.py
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
from src.preprocess_data import X_train_vec, X_val_vec, y_train, y_val, X_test_vec, y_test

# 定义模型
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, activation='relu', solver='adam', random_state=42)

# 训练模型
model.fit(X_train_vec, y_train)

# 预测验证集
y_val_pred = model.predict(X_val_vec)

# 计算并打印准确率
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Accuracy: {val_accuracy}")

# 预测测试集
y_test_pred = model.predict(X_test_vec)

# 计算并打印测试集准确率
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy}")

# 确保 model 目录存在
os.makedirs('data', exist_ok=True)

# 保存模型
joblib.dump(model, 'data/sentiment_model.pkl')

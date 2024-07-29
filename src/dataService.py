import asyncio
import json
import requests
import websockets
import joblib
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.DEBUG)

# 加载模型和矢量化器
model = joblib.load('data/sentiment_model.pkl')
_, _, _, _, _, _, vectorizer = joblib.load('data/preprocessed_data.pkl')

# 定义发送聊天记录到 Spring Boot 后端的方法
def send_chat_history_to_springboot(user_id, userMessage,botMessage):
    if user_id is None:
        raise ValueError("user_id cannot be None")

    url = 'http://localhost:8080/chathistories'
    headers = {'Content-Type': 'application/json'}
    chat_message = {
        'userMessage': userMessage,
        'botMessage': botMessage,
        'timestamp': datetime.now().isoformat(),
        'user_id': user_id
    }
    response = requests.post(url, json=chat_message, headers=headers)
    if response.status_code == 201:
        logging.debug("Chat history successfully sent to Spring Boot.")
    else:
        logging.error(f"Failed to send chat history to Spring Boot. Status code: {response.status_code}")


# 处理 WebSocket 连接
async def handle_connection(websocket, path):
    logging.debug("Client connected")
    try:
        async for message in websocket:
            # 解析 JSON 消息
            chat_message = json.loads(message)
            user_message = chat_message.get('UserMessage', '')
            user_id = chat_message.get('UserId', '')
            print(user_id)

            # 处理消息
            new_reviews_vec = vectorizer.transform([user_message])
            prediction = model.predict(new_reviews_vec)[0]

            # 创建响应
            response = json.dumps({'prediction': prediction})
            await websocket.send(response)
            logging.debug("Message processed and sent back")

            # 创建并发送聊天记录到 Spring Boot 后端
            send_chat_history_to_springboot(user_id, user_message, prediction)

    except websockets.ConnectionClosed:
        logging.debug("Client disconnected")

# 主程序入口
async def main():
    async with websockets.serve(handle_connection, "0.0.0.0", 5000):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())

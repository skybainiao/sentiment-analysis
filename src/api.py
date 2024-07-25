import asyncio
import json
import logging

import joblib
import websockets


logging.basicConfig(level=logging.DEBUG)


model = joblib.load('data/sentiment_model.pkl')
_, _, _, _, _, _, vectorizer = joblib.load('data/preprocessed_data.pkl')


async def handle_connection(websocket, path):
    logging.debug("Client connected")
    try:
        async for message in websocket:
            # 处理消息
            new_reviews_vec = vectorizer.transform([message])
            prediction = model.predict(new_reviews_vec)[0]
            response = json.dumps({'prediction': prediction})
            await websocket.send(response)
            logging.debug("Message processed and sent back")
    except websockets.ConnectionClosed:
        logging.debug("Client disconnected")


async def main():
    async with websockets.serve(handle_connection, "0.0.0.0", 5000):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())

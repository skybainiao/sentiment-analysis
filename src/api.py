from flask import Flask
from flask_socketio import SocketIO, send
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

model = joblib.load('data/sentiment_model.pkl')
_, _, _, _, _, _, vectorizer = joblib.load('data/preprocessed_data.pkl')

@app.route('/')
def index():
    return 'WebSocket server is running!'

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('message')
def handle_message(msg):
    new_reviews_vec = vectorizer.transform([msg])
    prediction = model.predict(new_reviews_vec)[0]
    send({'prediction': prediction}, json=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)

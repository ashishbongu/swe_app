import os
import numpy as np
import pandas as pd
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from io import BytesIO
import base64
import requests
import joblib
from transformers import pipeline

app = Flask(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force CPU, avoid CUDA errors

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

def get_stock_data(stock_symbol, years=4):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=365 * years)
    df = yf.download(stock_symbol, start=start, end=end, progress=False)
    if df.empty:
        return None
    df['MA20'] = df['Close'].rolling(20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(50, min_periods=1).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].diff().rolling(14).mean()))
    df['Momentum'] = df['Close'].pct_change(5)
    df['Volatility'] = df['Close'].rolling(20).std()
    return df.dropna()

def preprocess_data(data, seq_length=60):
    features = ['Close', 'Volume', 'MA20', 'MA50', 'RSI', 'Momentum', 'Volatility']
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data[features])
    X, y = [], []
    for i in range(len(scaled) - seq_length):
        X.append(scaled[i:i+seq_length])
        y.append(scaled[i+seq_length, 0])
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(32, return_sequences=True),
        Dropout(0.1),
        LSTM(16),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def predict_future(model, last_seq, scaler, n_days=30):
    preds = []
    current = last_seq.copy()
    for _ in range(n_days):
        pred = model.predict(current[np.newaxis, ...], verbose=0)[0, 0]
        new_row = current[-1].copy()
        new_row[0] = pred
        new_row[1:] *= 0.99
        current = np.vstack([current[1:], new_row])
        preds.append(pred)
    dummy = np.zeros((len(preds), 7))
    dummy[:, 0] = preds
    return scaler.inverse_transform(dummy)[:, 0]

def generate_plot(actual, predicted, future):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(actual)), actual, label='Actual')
    plt.plot(range(len(actual)-len(predicted), len(actual)), predicted, '--', label='Predicted')
    plt.plot(range(len(actual), len(actual)+len(future)), future, ':', label='Future')
    plt.legend()
    plt.grid()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()

@app.route('/')
def home():
    return '''
    <html><head><title>Stock Predictor</title></head>
    <body style="text-align:center;">
        <h1>Stock Predictor</h1>
        <form method="POST" action="/predict">
            <label>Symbol: <input type="text" name="symbol" required></label><br><br>
            <label>Days: <input type="number" name="days" value="30" min="1" max="90" required></label><br><br>
            <button type="submit">Predict</button>
        </form>
    </body></html>
    '''

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        stock_symbol = request.form.get('symbol', '').upper()
        pred_days = int(request.form.get('days', 30))
    else:
        data = request.get_json() or {}
        stock_symbol = data.get('symbol', '').upper()
        pred_days = int(data.get('days', 30))

    if not stock_symbol:
        return jsonify({'error': 'Symbol required'}), 400

    data = get_stock_data(stock_symbol)
    if data is None:
        return jsonify({'error': 'Invalid stock symbol'}), 400

    X, y, scaler = preprocess_data(data)
    model_path = os.path.join(MODEL_DIR, f"{stock_symbol}_model.h5")
    scaler_path = os.path.join(MODEL_DIR, f"{stock_symbol}_scaler.gz")

    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
        else:
            model = build_lstm_model(X.shape[1:])
            model.fit(X, y, epochs=3, batch_size=32, verbose=0)
            model.save(model_path)
            joblib.dump(scaler, scaler_path)

        future = predict_future(model, X[-1], scaler, pred_days)
        plot_url = generate_plot(data['Close'][-100:], y[-100:], future)

        if request.method == 'POST':
            return f"""
            <h2>{stock_symbol} Predictions</h2>
            <p>{future.tolist()}</p>
            <img src="data:image/png;base64,{plot_url}"><br><a href="/">Back</a>
            """
        return jsonify({'predictions': future.tolist(), 'graph': plot_url})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/company_info')
def company_info():
    symbol = request.args.get('symbol', '').upper()
    ticker = yf.Ticker(symbol)
    info = ticker.info
    return jsonify({
        'name': info.get('shortName', 'N/A'),
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A'),
        'market_cap': info.get('marketCap', 'N/A'),
        '52_week_range': f"{info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}"
    })

HF_TOKEN = os.getenv("HF_TOKEN")

def get_company_info(symbol):
    ticker = yf.Ticker(symbol)
    info = ticker.info
    news = []
    try:
        raw_news = ticker.news or []
        news = [n['title'] for n in raw_news[:5] if 'title' in n]
    except:
        pass
    if not news:
        name = info.get('shortName', symbol)
        sector = info.get('sector', 'Unknown')
        news = [
            f"{name} earnings expected.",
            f"{name} expands in {sector}.",
            f"Analysts watching {name} closely."
        ]
    return {
        "name": info.get('shortName', symbol),
        "sector": info.get('sector', 'Unknown'),
        "industry": info.get('industry', 'Unknown'),
        "news": news
    }

def get_sentiment_from_api(text):
    api_url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    try:
        r = requests.post(api_url, headers=headers, json={"inputs": text})
        if r.status_code == 200:
            result = r.json()
            return result[0] if isinstance(result, list) else {"label": "Neutral", "score": 0.0}
    except:
        pass
    return {"label": "Neutral", "score": 0.0}

@app.route('/sentiment')
def sentiment():
    symbol = request.args.get('symbol', '').upper()
    info = get_company_info(symbol)
    results = []
    for headline in info['news']:
        analysis = get_sentiment_from_api(headline)
        results.append({
            'headline': headline,
            'sentiment': analysis['label'],
            'score': analysis['score']
        })
    return jsonify({'sentiment_analysis': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))

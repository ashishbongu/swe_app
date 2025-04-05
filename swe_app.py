import os
import io
import joblib
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import datetime
from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

app = Flask(__name__)

MODEL_DIR = "models"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def get_model_path(symbol):
    return os.path.join(MODEL_DIR, f"{symbol}_model.keras")

def get_scaler_path(symbol):
    return os.path.join(MODEL_DIR, f"{symbol}_scaler.save")

def load_stock_data(symbol):
    end = datetime.datetime.today()
    start = end - datetime.timedelta(days=365*2)
    df = yf.download(symbol, start=start, end=end)
    return df[['Close']]

def preprocess_data(data, sequence_length=50):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])

    return np.array(X), np.array(y), scaler

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

@app.route('/')
def index():
    return render_template_string('''
        <h2>Stock Prediction</h2>
        <form action="/predict" method="post">
            Stock Symbol: <input type="text" name="symbol"><br>
            Days to Predict: <input type="number" name="days"><br>
            <input type="submit" value="Predict">
        </form>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        data = request.get_json()
        symbol = data['symbol']
        days_to_predict = int(data['days'])
    else:
        symbol = request.form['symbol']
        days_to_predict = int(request.form['days'])

    df = load_stock_data(symbol)
    X, y, scaler = preprocess_data(df.values)

    model_path = get_model_path(symbol)
    scaler_path = get_scaler_path(symbol)

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
    else:
        model = build_model((X.shape[1], 1))
        early_stop = EarlyStopping(monitor='loss', patience=2)
        model.fit(X, y, epochs=3, batch_size=32, verbose=0, callbacks=[early_stop])
        model.save(model_path)
        joblib.dump(scaler, scaler_path)

    last_sequence = X[-1]
    predictions = []
    for _ in range(days_to_predict):
        pred = model.predict(np.expand_dims(last_sequence, axis=0), verbose=0)[0]
        predictions.append(pred[0])
        last_sequence = np.append(last_sequence[1:], [[pred[0]]], axis=0)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

    plt.figure(figsize=(8, 4))
    plt.plot(predictions, label='Predicted')
    plt.title(f'{symbol} Stock Price Prediction')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_data = base64.b64encode(buf.getvalue()).decode('utf8')
    plt.close()

    if request.is_json:
        return jsonify({"predictions": predictions.tolist()})
    else:
        return render_template_string('''
            <h2>Prediction for {{symbol}}</h2>
            <img src="data:image/png;base64,{{graph}}"/>
            <br><a href="/">Back</a>
        ''', symbol=symbol.upper(), graph=graph_data)

@app.route('/sentiment', methods=['POST'])
def sentiment():
    data = request.get_json()
    text = data['text']
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    result = sentiment_pipeline(text)
    return jsonify(result)

@app.route('/company_info', methods=['GET'])
def company_info():
    symbol = request.args.get('symbol')
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return jsonify({
            "longName": info.get("longName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "summary": info.get("longBusinessSummary", "No summary available."),
        })
    except:
        return jsonify({"error": "Could not fetch company info."}), 500

if __name__ == '__main__':
    app.run(debug=True)

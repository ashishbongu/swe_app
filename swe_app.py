from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from transformers import pipeline



app = Flask(__name__)

# Function to fetch stock data
def get_stock_data(stock_symbol, years=4):
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=365 * years)
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    
    if data.empty:
        return None
    
    data['MA20'] = data['Close'].rolling(window=20, min_periods=1).mean()
    data['MA50'] = data['Close'].rolling(window=50, min_periods=1).mean()
    data['RSI'] = 100 - (100 / (1 + data['Close'].diff().rolling(14).mean()))
    data['Momentum'] = data['Close'].pct_change(periods=5)
    data['Volatility'] = data['Close'].rolling(window=20).std()
    
    return data.dropna()

# Function to preprocess data
def preprocess_data(data, seq_length=60):
    feature_cols = ['Close', 'Volume', 'MA20', 'MA50', 'RSI', 'Momentum', 'Volatility']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[feature_cols])
    
    X, y = [], []
    for i in range(len(scaled_data) - seq_length):
        X.append(scaled_data[i:i+seq_length])
        y.append(scaled_data[i+seq_length, 0])
    
    return np.array(X), np.array(y), scaler

# Function to build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Function to predict stock prices
def predict_future(model, last_sequence, scaler, n_days=30):
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(n_days):
        pred = model.predict(current_seq[np.newaxis, ...])[0, 0]
        new_row = current_seq[-1].copy()
        new_row[0] = pred
        new_row[1:] *= 0.99
        current_seq = np.vstack([current_seq[1:], new_row])
        predictions.append(pred)
    
    dummy = np.zeros((len(predictions), 7))
    dummy[:, 0] = predictions
    predictions = scaler.inverse_transform(dummy)[:, 0]
    
    return predictions

# Function to generate graph
def generate_plot(actual, predicted, future):
    plt.figure(figsize=(10, 5))

    # Debugging: Print lengths of data
    print(f"Actual Length: {len(actual)}, Predicted Length: {len(predicted)}, Future Length: {len(future)}")

    if len(actual) == 0 or len(predicted) == 0 or len(future) == 0:
        print("Error: One of the input lists is empty.")
        return None

    plt.plot(range(len(actual)), actual, label='Actual Price', color='blue')
    plt.plot(range(len(actual) - len(predicted), len(actual)), predicted, '--', label='Predicted', color='orange')
    plt.plot(range(len(actual), len(actual) + len(future)), future, ':', label='Future Prediction', color='red')

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Stock Price Prediction")
    plt.grid()

    # Save plot as Base64
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode()


    

# Endpoint for company info
@app.route('/company_info', methods=['GET'])
def company_info():
    stock_symbol = request.args.get('symbol', '').upper()
    if not stock_symbol:
        return jsonify({'error': 'Stock symbol is required'}), 400
    
    company = yf.Ticker(stock_symbol)
    info = company.info
    return jsonify({
        'name': info.get('shortName', 'N/A'),
        'sector': info.get('sector', 'N/A'),
        'industry': info.get('industry', 'N/A'),
        'market_cap': info.get('marketCap', 'N/A'),
        '52_week_range': f"{info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}"
    })

# Endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    stock_symbol = data.get('symbol', '').upper()
    pred_days = data.get('days', 30)
    
    stock_data = get_stock_data(stock_symbol)
    if stock_data is None:
        return jsonify({'error': 'Invalid stock symbol or no data available'}), 400
    
    X, y, scaler = preprocess_data(stock_data)
    model = build_lstm_model(X.shape[1:])
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    
    future_predictions = predict_future(model, X[-1], scaler, pred_days)
    plot_url = generate_plot(stock_data['Close'][-100:], y[-100:], future_predictions)
    
    return jsonify({
        'predictions': future_predictions.tolist(),
        'graph': plot_url
    })

# Sentiment analysis endpoint
# Function to get company info and fallback news (from your template)
def get_company_info(stock_symbol):
    try:
        company = yf.Ticker(stock_symbol)
        info = company.info

        # Try to get recent news
        news = []
        try:
            news_data = company.news
            if news_data and len(news_data) > 0:
                for item in news_data[:5]:  # Limit to 5
                    if 'title' in item:
                        news.append(item['title'])
        except:
            pass

        # Fallback headlines
        if not news:
            name = info.get('shortName', stock_symbol)
            sector = info.get('sector', 'Unknown')
            news = [
                f"{name} quarterly earnings expected next month.",
                f"{name} expanding operations in {sector} sector.",
                f"Analysts update ratings for {name}."
            ]

        return {
            "name": info.get('shortName', stock_symbol),
            "sector": info.get('sector', 'Unknown'),
            "industry": info.get('industry', 'Unknown'),
            "news": news
        }
    except Exception as e:
        print(f"Error fetching company info: {e}")
        return {
            "name": stock_symbol,
            "sector": "Unknown",
            "industry": "Unknown",
            "news": [
                f"{stock_symbol} quarterly earnings expected next month.",
                f"{stock_symbol} market conditions changing.",
                f"Investors watching {stock_symbol} closely."
            ]
        }

# Updated sentiment analysis endpoint using get_company_info
@app.route('/sentiment', methods=['GET'])
def sentiment():
    stock_symbol = request.args.get('symbol', '').upper()
    info = get_company_info(stock_symbol)
    headlines = info['news']

    sentiment_pipeline = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    results = []

    for headline in headlines:
        sentiment = sentiment_pipeline(headline)[0]
        results.append({
            'headline': headline,
            'sentiment': sentiment['label'],
            'score': sentiment['score']
        })

    return jsonify({'sentiment_analysis': results})


if __name__ == '__main__':
    app.run(debug=True)

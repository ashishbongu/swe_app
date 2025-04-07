import json
from flask import send_file
import io
import os
import matplotlib
from flask import Flask, render_template, request
from lstm_model import LSTMModel
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, ParameterGrid
import pandas as pd
from datetime import datetime, timedelta
matplotlib.use('Agg')
import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
BEST_MODEL_PATH = 'best_model.pth'
BEST_PARAMS_PATH = 'best_params.json'

API_KEY = os.getenv("NEWSDATA_API_KEY")

def fetch_headlines_from_newsapi(ticker):
    url = f"https://newsdata.io/api/1/news?apikey={API_KEY}&q={ticker}&language=en&category=business"
    response = requests.get(url)
    data = response.json()

    headlines = []
    
    if "results" in data:
        for item in data["results"]:
            title = item.get("title")
            if title:
                headlines.append(title)

    return headlines[:10]


    
def analyze_sentiment(headlines):
    url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
    headers = {
        "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"
    }

    results = []
    for headline in headlines:
        response = requests.post(url, headers=headers, json={"inputs": headline})
        if response.status_code == 200:
            output = response.json()[0]
            top_label = max(output, key=lambda x: x['score'])['label']
            results.append({"headline": headline, "sentiment": top_label})
        else:
            results.append({"headline": headline, "sentiment": "Error"})
    return results







def add_moving_average_features(df, window_sizes=[5, 10, 20]):
    for window in window_sizes:
        df[f'MA_{window}'] = df['Close'].rolling(window=window).mean()
    df.dropna(inplace=True)
    return df


def train_evaluate_model(x_train, y_train, x_test, y_test, params, device='cpu', num_epochs=100):
    model = LSTMModel(input_dim=x_train.shape[2],
                      hidden_dim=params['hidden_dim'],
                      num_layers=params['num_layers'],
                      output_dim=params['output_dim'],
                      dropout=params['dropout']).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.MSELoss()

    best_loss = np.inf
    best_model_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train.to(device))
        loss = criterion(outputs, y_train.to(device))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            test_outputs = model(x_test.to(device))
            test_loss = criterion(test_outputs, y_test.to(device))

        if test_loss < best_loss:
            best_loss = test_loss
            best_model_state = model.state_dict()

        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {loss.item():.6f}, Test Loss: {test_loss:.6f}')

    return best_loss.item(), best_model_state


@app.route("/sentiment", methods=["GET", "POST"])
def sentiment():
    sentiments = []
    ticker = ""
    if request.method == "POST":
        ticker = request.form["ticker"]
        headlines = fetch_headlines_from_newsapi(ticker)
        print("Fetched headlines:", headlines)  # Optional debug
        if headlines:
            sentiments = analyze_sentiment(headlines)
            print("Sentiments:", sentiments)  # Optional debug
    return render_template("sentiment.html", sentiments=sentiments, ticker=ticker)



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/prediction')
def prediction():
    return render_template('prediction.html')

@app.route('/download_predictions')
def download_predictions():
    return send_file("static/predictions.csv", as_attachment=True)

@app.route('/company-info')
def company_info():
    return render_template('company-info.html')


@app.route('/predict', methods=['POST'])
def predict():

    ticker = request.form['ticker'].upper()
    stock = yf.Ticker(ticker)

    company_info = {
        'longName': stock.info.get('longName', 'N/A'),
        'symbol': ticker,
        'sector': stock.info.get('sector', 'N/A'),
        'industry': stock.info.get('industry', 'N/A'),
        'website': stock.info.get('website', '#'),
        'summary': stock.info.get('longBusinessSummary', 'N/A')
    }

    tickerSymbol = request.form['ticker']
    recent_update_date = request.form['date']
    n_days = int(request.form.get('n_days', 5))

    tickerData = yf.Ticker(tickerSymbol)
    df = tickerData.history(period='1d', start='2010-01-01', end=recent_update_date)
    df = add_moving_average_features(df)

    data = df[['Close', 'MA_5', 'MA_10', 'MA_20']].values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)

    max_original = df['Close'].max()
    min_original = df['Close'].min()

    sequence_length = 60
    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i - sequence_length:i, :-1])
        y.append(scaled_data[i, -1])
    x, y = np.array(x), np.array(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    if os.path.exists(BEST_MODEL_PATH) and os.path.exists(BEST_PARAMS_PATH):
        best_params = json.load(open(BEST_PARAMS_PATH))
        model = LSTMModel(input_dim=x_train.shape[2],
                          hidden_dim=best_params['hidden_dim'],
                          num_layers=best_params['num_layers'],
                          output_dim=best_params['output_dim'],
                          dropout=best_params['dropout']).to(device)
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
    else:
        param_grid = {
            'hidden_dim': [64, 128],
            'num_layers': [2, 3],
            'output_dim': [1],
            'dropout': [0.2, 0.3],
            'lr': [0.001, 0.0001]
        }

        best_loss = np.inf
        best_params = None
        best_model_state = None

        for params in ParameterGrid(param_grid):
            print(f'Training with parameters: {params}')
            loss, state = train_evaluate_model(x_train, y_train, x_test, y_test, params, device=device.__str__())
            if state is not None and (best_model_state is None or loss < best_loss):
                best_loss = loss
                best_params = params
                best_model_state = state

        if best_model_state is None:
            return "Training failed."

        torch.save(best_model_state, BEST_MODEL_PATH)
        with open(BEST_PARAMS_PATH, 'w') as f:
            json.dump(best_params, f)

        model = LSTMModel(input_dim=x_train.shape[2],
                          hidden_dim=best_params['hidden_dim'],
                          num_layers=best_params['num_layers'],
                          output_dim=best_params['output_dim'],
                          dropout=best_params['dropout']).to(device)
        model.load_state_dict(best_model_state)
        model.eval()

    with torch.no_grad():
        test_outputs = model(x_test.to(device))

    test_outputs_np = test_outputs.cpu().numpy().reshape(-1)
    predictions_rescaled = test_outputs_np * (max_original - min_original) + min_original

    # 1️⃣ Graph: Actual vs Predicted
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[-len(predictions_rescaled):], df['Close'].values[-len(predictions_rescaled):],
             label='Actual Prices')
    plt.plot(df.index[-len(predictions_rescaled):], predictions_rescaled, label='Predicted Prices')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plot_path = 'static/prediction_plot.png'
    plt.savefig(plot_path)
    plt.close()

    # 2️⃣ Graph: Strategy Backtesting
    signals = np.diff(predictions_rescaled) > 0
    signals = np.insert(signals, 0, False)
    actual = df['Close'].values[-len(predictions_rescaled):]
    daily_returns = np.diff(actual) / actual[:-1]
    daily_returns = np.insert(daily_returns, 0, 0)
    strategy_returns = signals[:-1] * daily_returns[1:]
    cumulative_strategy_returns = np.cumsum(strategy_returns)
    cumulative_stock_returns = np.cumsum(daily_returns)
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_stock_returns, label='Stock Returns (Buy and Hold)')
    plt.plot(cumulative_strategy_returns, label='Strategy Returns')
    plt.title('Backtesting Stock Price Prediction Strategy')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Returns')
    plt.legend()
    plot_path1 = 'static/backtesting_plot.png'
    plt.savefig(plot_path1)
    plt.close()

    # 3️⃣ Graph: Future Forecast
    last_sequence = scaled_data[-sequence_length:, :-1]
    input_seq = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    predicted_future = []

    for _ in range(n_days):
        with torch.no_grad():
            next_pred = model(input_seq)
        next_price_scaled = next_pred.item()
        predicted_future.append(next_price_scaled)

        new_step = input_seq[0, -1, :].cpu().numpy()
        new_step = np.append(new_step, next_price_scaled)
        new_step_features = new_step[:-1]
        input_seq = input_seq.squeeze(0).cpu().numpy()
        input_seq = np.vstack([input_seq[1:], new_step_features])
        input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)

    today_dt = df.index[-1].to_pydatetime()

# Future predictions (rescale)
    future_preds_rescaled = np.array(predicted_future) * (max_original - min_original) + min_original

# Generate future dates (skipping weekends)
    future_dates = []
    next_day = today_dt
    while len(future_dates) < n_days:
        next_day += timedelta(days=1)
        if next_day.weekday() < 5:  # Weekdays only
            future_dates.append(next_day)

    prediction_table = [(date.strftime('%Y-%m-%d'), round(price, 2)) for date, price in zip(future_dates, future_preds_rescaled)]
                        
    plt.figure(figsize=(12, 6))
    plt.plot(future_dates, future_preds_rescaled, marker='o', linestyle='--', color='green', label=f'{n_days}-Day Forecast')
    plt.title(f'{tickerSymbol.upper()} {n_days}-Day Stock Price Forecast (from {today_dt.strftime("%Y-%m-%d")})')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plot_path2 = 'static/future_plot.png'
    plt.savefig(plot_path2)
    plt.close()
    
    # Save prediction table to CSV in memory
    csv_buffer = io.StringIO()
    pd.DataFrame(prediction_table, columns=["Date", "Predicted Price"]).to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    csv_data = csv_buffer.getvalue()

# Save CSV to a temporary file or keep in memory for download
    with open("static/predictions.csv", "w") as f:
        f.write(csv_data)

    
    
    return render_template('prediction.html',
                           plot_path=plot_path,
                           plot_path1=plot_path1,
                           plot_path2=plot_path2,
                           prediction_table=prediction_table,
                           company_info=company_info)


if __name__ == '__main__':
    app.run(debug=True)

# Tradesecret - Stock Price Forecasting and Sentiment Analyizer

This repository contains a Flask web application for predicting stock prices using Long Short-Term Memory (LSTM) neural networks. The application fetches historical stock data from Yahoo Finance API (`yfinance`), preprocesses the data by adding moving average features, normalizes it using Min-Max scaling, and trains LSTM models with different hyperparameters and also analyzes the sentimment analysis of latest new headlines of a chosen company using ***finBert LLM model***.


## Group Members
- **23BDS014 - BONGU ASHISH**
- **23BDS041 - PB SHREYAS**
- **23BDS062 - TARAN JAIN**
- **23BDS016 - CHAITRA V KATTIMANI** 
- **23BDS027 - KANISHK PANDEY** 
- **23BDS0XX - ISHAN**


## Tech Stack

| Layer            | Technology                         |
|------------------|-------------------------------------|
| Backend          | Flask, Python, REST API             |
| Frontend         | HTML, CSS, Jinja2, Bootstrap        |
| Machine Learning | PyTorch, LSTM, LLM ([FinBERT](https://huggingface.co/ProsusAI/finbert)) |
| News API         | NewsData.io                         |
| Data             | Yahoo Finance (yfinance)            |
| Deployment       | Localhost, Render(limited access)   |


## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/ChellaVigneshKP/stock-prediction.git
   ```
2. Start the Flask application:
   ```bash
   python app.py
   ```
3. Navigate to http://localhost:5000 in your web browser.
4. Enter a stock ticker symbol and a recent date to get predictions and backtesting results.

## Screeshots
![image](https://github.com/ChellaVigneshKP/stock-prediction/assets/97314418/964acfc0-5e28-4cee-b7df-682af6996665)
![image](https://github.com/ChellaVigneshKP/stock-prediction/assets/97314418/95f5e097-733f-4941-b4e6-cfc4eae74d35)



import json
import os

from flask import Flask, render_template, request
from lstm_model import LSTMModel
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
BEST_MODEL_PATH = 'best_model.pth'
BEST_PARAMS_PATH = 'best_params.json'


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
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {loss.item():.6f}, Test Loss: {test_loss.item():.6f}')

    return best_loss.item(), best_model_state


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    tickerSymbol = request.form['ticker']
    recent_update_date = request.form['date']
    tickerData = yf.Ticker(tickerSymbol)
    df = tickerData.history(period='1d', start='2010-01-01', end=recent_update_date)
    print(f"Using device: {device}")

    def normalize_data(data):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(data)
        return scaler, scaled_data

    def prepare_sequences(data, sequence_length):
        x_data, y_data = [], []
        for i in range(sequence_length, len(data)):
            x_data.append(data[i - sequence_length:i, :-1])
            y_data.append(data[i, -1])
        return np.array(x_data), np.array(y_data)

    df = add_moving_average_features(df)
    data = df[['Close', 'MA_5', 'MA_10', 'MA_20']].values
    scaler, scaled_data = normalize_data(data)

    sequence_length = 60
    x, y = prepare_sequences(scaled_data, sequence_length)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

    # Convert to PyTorch tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    if os.path.exists(BEST_MODEL_PATH) and os.path.exists(BEST_PARAMS_PATH):
        # Load the best model and parameters if they exist
        best_params = json.load(open(BEST_PARAMS_PATH))
        model = LSTMModel(input_dim=x_train.shape[2],  # Adjust input_dim according to your feature size
                          hidden_dim=best_params['hidden_dim'],
                          num_layers=best_params['num_layers'],
                          output_dim=best_params['output_dim'],
                          dropout=best_params['dropout']).to(device)
        model.load_state_dict(torch.load(BEST_MODEL_PATH,map_location=torch.device('cpu')))
        model.eval()
    else:
        # Train a new model if no best model is found
        param_grid = {
            'hidden_dim': [64, 128],
            'num_layers': [2, 3],
            'output_dim': [1],
            'dropout': [0.2, 0.3],
            'lr': [0.001, 0.0001]
        }

        best_test_loss = np.inf
        best_params = None
        best_model_state = None

        for params in ParameterGrid(param_grid):
            print(f'Training with parameters: {params}')
            test_loss, current_best_model_state = train_evaluate_model(x_train, y_train, x_test, y_test, params,
                                                                       device=device.__str__())
            if current_best_model_state is not None and (best_model_state is None or test_loss < best_test_loss):
                best_test_loss = test_loss
                best_params = params
                best_model_state = current_best_model_state

        if best_model_state is None:
            return "No valid model state found. Training did not succeed."

        print(f'\nBest parameters: {best_params}')
        print(f'Best test loss: {best_test_loss:.6f}')

        # Save the best model and parameters
        torch.save(best_model_state, BEST_MODEL_PATH)
        with open(BEST_PARAMS_PATH, 'w') as params_file:
            json.dump(best_params, params_file)

        # Load the best model for predictions
        model = LSTMModel(input_dim=x_train.shape[2],  # Adjust input_dim according to your feature size
                          hidden_dim=best_params['hidden_dim'],
                          num_layers=best_params['num_layers'],
                          output_dim=best_params['output_dim'],
                          dropout=best_params['dropout']).to(device)
        model.load_state_dict(best_model_state)
        model.eval()

        # Perform predictions on test set
    with torch.no_grad():
        test_outputs = model(x_test.to(device))

    # predictions_rescaled = scaler.inverse_transform(test_outputs.cpu().numpy()).reshape(-1)
    max_original = df['Close'].max()
    min_original = df['Close'].min()
    test_outputs_np = test_outputs.cpu().numpy().reshape(-1)
    predictions_rescaled = test_outputs_np * (max_original - min_original) + min_original
    print("Shape of predictions_rescaled:", predictions_rescaled.shape)

    # Plotting results
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

    signals = np.diff(predictions_rescaled) > 0
    signals = np.insert(signals, 0, False)

    # Calculate daily returns of the stock
    actual = df['Close'].values[-len(predictions_rescaled):]
    rmse = np.sqrt(np.mean((predictions_rescaled - actual) ** 2))
    # Print the RMSE
    print(f'Test RMSE: {rmse}')
    daily_returns = np.diff(actual) / actual[:-1]
    daily_returns = np.insert(daily_returns, 0, 0)

    # Calculate strategy returns
    strategy_returns = signals[:-1] * daily_returns[1:]

    # Calculate cumulative returns for the strategy and the stock (buy and hold)
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
    # Render prediction result with plot on HTML template
    return render_template('prediction_result.html', plot_path=plot_path, plot_path1=plot_path1)


if __name__ == '__main__':
    app.run(debug=True)

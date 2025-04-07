# Tradesecret - Stock Price Forecasting and Sentiment Analyizer

This repository contains a Flask web application for predicting stock prices using Long Short-Term Memory (LSTM) neural networks. The application fetches historical stock data from Yahoo Finance API (`yfinance`), preprocesses the data by adding moving average features, normalizes it using Min-Max scaling, and trains LSTM models with different hyperparameters and also analyzes the sentimment analysis of latest new headlines of a chosen company using ***finBert LLM model***.


## Group Members
- **23BDS014 - BONGU ASHISH**
- **23BDS041 - PB SHREYAS**
- **23BDS062 - TARAN JAIN**
- **23BDS016 - CHAITRA V KATTIMANI** 
- **23BDS027 - KANISHK PANDEY** 
- **23BDS023 - ISHAN SRIVASTAVA**


## Tech Stack

| Layer            | Technology                         |
|------------------|-------------------------------------|
| Backend          | Flask, Python, REST API             |
| Frontend         | HTML, CSS, Jinja2, Bootstrap        |
| Machine Learning | PyTorch, LSTM, LLM ([FinBERT](https://huggingface.co/ProsusAI/finbert)) |
| News API         | NewsData.io                         |
| Data             | Yahoo Finance (yfinance)            |
| Deployment       | Localhost, Render(limited access)   |

## FOLDER STRUCTURE

<pre lang="nohighlight"><code> 
.
├── .env
├── .gitignore
├── app.py
├── best_model.pth
├── best_params.json
├── CNAME
├── lstm_model.py
├── README.md
│
├── static
│   ├── backtesting_plot.png
│   ├── feedback.png
│   ├── forecast.png
│   ├── future_plot.png
│   ├── info.png
│   ├── logo.png
│   ├── pngegg.png
│   ├── predictions.csv
│   └── styles.css
│
├── templates
│   ├── company-info.html
│   ├── index.html
│   ├── login.html
│   ├── prediction.html
│   ├── prediction_result.html
│   └── sentiment.html
│
├── web development
│   └── index.html
│
└── __pycache__
    └── lstm_model.cpython-312.pyc

</code></pre>



## How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/ashishbongu/SWE_Project.git
   ```
2. Locate the Folder (may change depending on your files management):
   ```bash
   cd SWE_Project
   ```
3. Install all the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Create API KEYS from https://huggingface.co/settings/tokens and https://newsdata.io/api-key (Note: Make sure to create Account).
   
5. Create a (.env) file in main project folder and paste this code by replacing with your API Keys (use vs code for ease) 
   ```bash
   HUGGINGFACE_API_KEY=your_huggingface_api_key
   NEWSDATA_API_KEY=your_newsdata_api_key
   ```
6. Run the Flask App:
   ```bash
   python app.py
   ```
   It will display this and there you go !! Ctrl+click on the link to navigate to local server
![Screenshot 2025-04-07 101733](https://github.com/user-attachments/assets/8c3ab48e-b204-4a67-8f03-bf508c94f822)

 
## SNAPSHOTS OF OUR WEB APPLICATION - TRADE SECRET

![Screenshot 2025-04-07 080200](https://github.com/user-attachments/assets/40069881-440b-4010-8028-3061c6bccccd)
![Screenshot 2025-04-07 080221](https://github.com/user-attachments/assets/08cde385-127f-4416-999b-13ba324ca581)
![Screenshot 2025-04-07 102452](https://github.com/user-attachments/assets/beb93acb-afc7-4380-a295-125e61f63f88)
![Screenshot 2025-04-07 102555](https://github.com/user-attachments/assets/1610fc21-f347-4967-894b-ff33f74ded93)
![image](https://github.com/user-attachments/assets/92d09ccd-7672-489a-91db-c2b994b69ae9)
![image](https://github.com/user-attachments/assets/2d1b8c63-917b-42a7-b939-6448ea0d595b)
![Screenshot 2025-04-07 102647](https://github.com/user-attachments/assets/6f7658f3-faea-4bd7-ada9-9acad7f5135a)









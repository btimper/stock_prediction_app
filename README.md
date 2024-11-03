# Stock Prediction App

This application provides a graphical user interface for predicting stock movements using historical data. It allows users to input a stock ticker and a number of days for prediction, and displays the prediction results along with a plot of the stock's price history.

## Features

- Input a stock ticker symbol
- Specify the number of days for prediction
- Display today's buy/sell prediction
- Show model precision over the last three months
- Visualize stock price history

## Installation

1. Clone this repository:

git clone https://github.com/btimper/stock-prediction-app.git
cd stock-prediction-app

2. Create a virtual environment (optional but recommended):

python -m venv .venv
source .venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install the required packages:

pip install yfinance pandas scikit-learn matplotlib Pillow


## Usage

Run the application using:

python stock_prediction_app.py

1. Enter a valid stock ticker (e.g., AAPL for Apple Inc.) in the "Enter Stock Ticker" field.
2. Enter the number of days for prediction in the "Enter Number of Days" field.
3. Click the "Predict" button to see the results.

The application will display today's prediction (Buy or Sell), the model's precision over the last three months, and a plot of the stock's price history.

## Dependencies

- Python 3.7+
- tkinter
- yfinance
- pandas
- scikit-learn
- matplotlib
- Pillow

## Acknowledgments

- This project uses the yfinance library to fetch stock data.
- The machine learning model is built using scikit-learn.
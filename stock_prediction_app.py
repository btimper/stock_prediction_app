import tkinter as tk
from tkinter import messagebox
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image, ImageTk

# Function to get data and create predictors
def getData(ticker, days):
    data = yf.Ticker(ticker)
    data = data.history(period="max")
    
    del data["Dividends"]
    del data["Stock Splits"]

    # Shift target column to predict future prices
    data["Target Day"] = data["Close"].shift(-days)
    data["Target"] = (data["Target Day"] > data["Close"]).astype(int)

    # Filter for dates after 1990
    data = data.loc["1990-01-01":].copy()

    horizons = [2,5,60,250,1000]
    predictors = []
    
    for horizon in horizons:
        rolling_averages = data.rolling(horizon).mean()
    
        ratio_column = f"Close_Ratio_{horizon}"
        data[ratio_column] = data["Close"] / rolling_averages["Close"]
    
        trend_column = f"Trend_{horizon}"
        data[trend_column] = data.shift(1).rolling(horizon).sum()["Target"]
        
        predictors += [ ratio_column, trend_column]

    today = data.tail(1)

    # Drop rows with missing values
    data = data.dropna()
    
    return data, predictors, today

# Function to predict on test set
def predict(data, model, predictors):
    train = data.iloc[:-60].copy()
    test = data.iloc[-60:].copy()

    model.fit(train[predictors], train["Target"])
    
    preds = model.predict_proba(test[predictors])[:,1]

    # Apply threshold for buy/sell decision
    preds[preds >= .6] = 1 # Buy signal
    preds[preds < .6] = 0 # Sell signal
    
    preds = pd.Series(preds, index=test.index, name="Predictions")
    
    combined = pd.concat([test["Target"], preds], axis=1)
    
    return combined, model

def predict_today(model, today, predictors):
    # Predict the probability of a "buy" signal (class 1) for today's data
    today_pred = model.predict_proba(today[predictors])[:, 1]

    print(today_pred)
    
    # Apply threshold for buy/sell decision (0.6 threshold used in previous function)
    if today_pred >= 0.6:
        return "Buy"
    else:
        return "Sell"
    
def create_plot(data):
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['Close'])
    plt.title('Stock Price History')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img

def predict_stock():
    ticker = ticker_entry.get()
    days = days_entry.get()
    
    try:
        days = int(days)  # Convert to integer
        data, predictors, today = getData(ticker, days)
        model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
        predictions, model = predict(data, model, predictors)
        precision = precision_score(predictions["Target"], predictions["Predictions"])
        today_decision = predict_today(model, today, predictors)
        
        output_text = f"Today's prediction: {today_decision}\nModel precision over last 3 months: {precision:.2f}"
        output_label.config(text=output_text)
        
        # Create and display the plot
        img = create_plot(data)
        img = Image.open(img)
        img = ImageTk.PhotoImage(img)
        
        plot_label.imgtk = img  # Keep a reference to avoid garbage collection
        plot_label.configure(image=img)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Create the main window
root = tk.Tk()
root.title("Stock Prediction App")

# Create input fields and labels
ticker_label = tk.Label(root, text="Enter Stock Ticker:")
ticker_label.pack()

ticker_entry = tk.Entry(root)
ticker_entry.pack()

days_label = tk.Label(root, text="Enter Number of Days:")
days_label.pack()

days_entry = tk.Entry(root)
days_entry.pack()

# Create buttons
predict_button = tk.Button(root, text="Predict", command=predict_stock)
predict_button.pack()

# Output label for displaying results
output_label = tk.Label(root, text="", wraplength=400)
output_label.pack()

# Label for displaying the plot
plot_label = tk.Label(root)
plot_label.pack()

# Start the GUI event loop
root.mainloop()

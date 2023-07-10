import requests
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import joblib
from datetime import datetime


def train_model(symbol):
    # Define start and end dates
    start_date = datetime(2016, 1, 1).strftime("%Y-%m-%d")
    end_date = datetime(2021, 12, 31).strftime("%Y-%m-%d")

    # Download historical data from Yahoo Finance
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    if stock_data.empty:
        # No data available
        return

    # Preprocess the data
    stock_data['Year'] = stock_data.index.year
    stock_data['Month'] = stock_data.index.month
    stock_data['Day'] = stock_data.index.day
    stock_data = stock_data[['Year', 'Month', 'Day', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Split the data into features and target variable
    X = stock_data.drop('Close', axis=1)
    y = stock_data['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = SVR()
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f"Model score: {score}")

    # Save the model to a file
    joblib.dump(model, 'predictor/models/model.joblib')



def predict_future_prices(symbol, start_date, end_date):
    # Load the trained model
    model = joblib.load('predictor/models/model.joblib')

    # Download historical data from Yahoo Finance for prediction
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    if len(stock_data) == 0:
        # No data available
        return []

    # Preprocess the data
    stock_data['Date'] = stock_data.index
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    stock_data['Year'] = stock_data['Date'].dt.year
    stock_data['Month'] = stock_data['Date'].dt.month
    stock_data['Day'] = stock_data['Date'].dt.day
    stock_data = stock_data[['Year', 'Month', 'Day', 'Open', 'High', 'Low', 'Close', 'Volume']]

    # Use the model to make predictions
    X = stock_data.drop('Close', axis=1)
    predictions = model.predict(X)

    # Return the predicted prices as a list
    return predictions if isinstance(predictions, list) else [predictions]

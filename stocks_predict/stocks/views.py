from django.shortcuts import render
import yfinance as yf
import pandas as pd
import joblib
import os

def load_model():
    model_path = os.path.abspath(r"C:\Users\tejal\OneDrive\Documents\Internship\stock_model1.joblib")
    # Load the saved model
    model = joblib.load(model_path)
    return model

def get_stock_data(ticker):
    data = yf.download(ticker, start="2016-01-01", end="2021-12-31")
    df = pd.DataFrame(data['Adj Close'])
    return df

def make_predictions(model, df):
    predictions = model.predict(df)
    return predictions

def predict_stock(request):
    model = load_model()

    stock_ticker = "AAPL"
    stock_data = get_stock_data(stock_ticker)
    print(stock_data.columns)
    
    future_dates = pd.date_range(start=stock_data.index[-1], periods=10, freq='D')
    future_features = pd.DataFrame(data={'Adj Close': [stock_data['Adj Close'].iloc[-1]]}, index=future_dates)
    future_predictions = model.predict(future_features)

    print("Future Predictions:")
    for date, prediction in zip(future_dates, future_predictions):
        print("{}: {}".format(date.strftime("%Y-%m-%d"), prediction))
        
    context = {'prediction': future_predictions}
    return render(request, 'predict_stock.html', context)

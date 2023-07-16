import yfinance as yf
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from django.shortcuts import render
#from sklearn.externals import joblib
import joblib
import pandas as pd


def train_model(symbol, duration):
    # Calculate the start and end dates based on the duration
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=duration*365)).strftime('%Y-%m-%d')

    # Download stock data from Yahoo Finance
    stock_data = yf.download(symbol, start=start_date, end=end_date)

    # Prepare the data for training
    stock_data['Date'] = stock_data.index
    stock_data['Date'] = stock_data['Date'].map(datetime.toordinal)

    X = stock_data[['Date']].values
    y = stock_data['Close'].values

    # Create and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Save the trained model to a file
    joblib.dump(model, 'trained_model.pkl')

def predict_future_prices(symbol, prediction_period):
    # Load the trained model from the file
    model = joblib.load('trained_model.pkl')

    # Calculate the start and end dates for prediction
    end_date = datetime.now().date() + timedelta(days=prediction_period)
    start_date = datetime.now().date() + timedelta(days=1)

    # Generate the future dates
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')

    # Convert future dates to ordinal values
    future_dates_ordinals = [date.toordinal() for date in future_dates]
    future_dates_ordinals = [[ordinal] for ordinal in future_dates_ordinals]

    # Predict the future prices
    future_prices = model.predict(future_dates_ordinals)

    return zip(future_dates, future_prices)


def stock_prediction(request):
    if request.method == 'POST':
        symbol = request.POST.get('symbol')
        duration = int(request.POST.get('duration'))
        prediction_period = int(request.POST.get('prediction_period'))

        # Train the model if it doesn't exist
        try:
            model = joblib.load('trained_model.pkl')
        except FileNotFoundError:
            train_model(symbol, duration)

        # Predict future prices
        future_prices = predict_future_prices(symbol, prediction_period)

        # Prepare the output data
        output_data = {
            'symbol': symbol,
            'duration': duration,
            'future_prices': future_prices,
        }

        return render(request, 'prediction.html', {'output_data': output_data})

    return render(request, 'prediction.html')

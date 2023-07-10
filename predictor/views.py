from django.shortcuts import render
import yfinance as yf
import datetime
import plotly.graph_objects as go
from plotly.offline import plot
import numpy as np
from sklearn.linear_model import LinearRegression

def stock_prediction(request):
    if request.method == 'POST':
        stock_symbol = request.POST['stock_symbol']
        num_days = int(request.POST['num_days'])
        future_days = int(request.POST['future_days'])
        start_date = datetime.datetime.now() - datetime.timedelta(days=num_days)
        end_date = datetime.datetime.now()
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

        if stock_data.empty:
            error_message = f"No data available for symbol: {stock_symbol}"
            return render(request, 'stocks/stock_prediction.html', {'error_message': error_message})

        closing_prices = stock_data['Close']

        # Prepare training data
        X_train = np.array([[i] for i in range(len(closing_prices))])
        y_train = closing_prices.values

        # Train linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Generate future dates
        future_dates = pd.date_range(start=end_date, periods=future_days)

        # Generate future prices with random slope
        last_price = closing_prices.iloc[-1]
        future_prices = [last_price]
        for i in range(1, future_days):
            random_slope = np.random.uniform(-1, 1)
            future_price = future_prices[i - 1] + random_slope
            future_prices.append(future_price)

        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=closing_prices.index, y=closing_prices, mode='lines', name='Historical Prices'))
        fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines', name='Predicted Prices'))
        fig.update_layout(
            title=f"Stock Price Prediction for {stock_symbol}",
            xaxis_title="Date",
            yaxis_title="Price",
            template='plotly_white'
        )

        plot_div = plot(fig, output_type='div')

        return render(request, 'stocks/stock_prediction.html', {'plot_div': plot_div})

    return render(request, 'stocks/stock_prediction.html')

import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.offline as opy
import pandas as pd
from django.shortcuts import render
import re
import csv
from time import sleep
from bs4 import BeautifulSoup
import requests
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from yahooquery import search
from statsmodels.tsa.arima.model import ARIMA
from textblob import TextBlob
from selenium import webdriver
from urllib.parse import urljoin
headers = {
    'accept': '*/*',
    'accept-encoding': 'gzip, deflate, br',
    'accept-language': 'en-US,en;q=0.9',
    'referer': 'https://www.google.com',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.83 Safari/537.36 Edg/85.0.564.44'
}

def train_model(symbol,duration):
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=duration * 365)).strftime('%Y-%m-%d')

    # Download stock data from Yahoo Finance
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    # splitting data into train and testing 

    data_training = pd.DataFrame(stock_data['Close'][0:int(len(stock_data)*0.70)])
    data_testing = pd.DataFrame(stock_data['Close'][int(len(stock_data)*0.70):int(len(stock_data))])

    scaler = MinMaxScaler(feature_range=(0,1))

    data_training_array = scaler.fit_transform(data_training)

    x_train = []
    y_train = []

    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i-100:i])
        y_train.append(data_training_array[i,0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    model = Sequential()
    model.add(LSTM(units = 50, activation = 'relu', return_sequences= True,
                input_shape= (x_train.shape[1],1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 60, activation= 'relu',return_sequences= True))
    model.add(Dropout(0.3))

    model.add(LSTM(units = 80, activation= 'relu',return_sequences= True))
    model.add(Dropout(0.4))

    model.add(LSTM(units = 120, activation= 'relu'))
    model.add(Dropout(0.5))

    model.add(Dense(units = 1))
    model.compile(optimizer='adam',loss = 'mean_squared_error')
    model.fit(x_train,y_train,epochs=50)
    model.save('keras_model.h5')

    past_100_days = data_training.tail(1825)
    data_testing = past_100_days.append(data_testing, ignore_index=True)
    input_data = scaler.fit_transform(data_testing)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100:i])
        y_test.append(input_data[i,0])

    x_test,y_test = np.array(x_test), np.array(y_test)

    y_predicted = model.predict(x_test)

    scale_factor = 1/scaler.scale_
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor
    y_test = data_testing.values  # Assuming data_testing contains the test data
    y_predicted = model.predict(x_test)
    print(y_predicted)
    print(y_test)
    return y_test, y_predicted

# Function to perform sentiment analysis on the article description
def get_sentiment(description):
    blob = TextBlob(description)
    sentiment = blob.sentiment
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity
    return polarity, subjectivity

def get_article(card):
    """Extract article information from the raw html"""
    headline = card.find('h4', 's-title').text
    source = card.find("span", 's-source').text
    description = card.find('p', 's-desc').text.strip()
    raw_link = card.find('a').get('href')
    unquoted_link = requests.utils.unquote(raw_link)
    pattern = re.compile(r'RU=(.+)\/RK')
    clean_link_match = re.search(pattern, unquoted_link)
    clean_link = clean_link_match.group(1) if clean_link_match else None

    # Perform sentiment analysis on the article description
    polarity, subjectivity = get_sentiment(description)

    article = (headline, source, description, clean_link,polarity,subjectivity)
    return article[:6]

def get_color_from_polarity(polarity):
    if polarity > 0:
        return 'green'  # Positive polarity
    elif polarity < 0:
        return 'red'    # Negative polarity
    else:
        return 'gray'   # Neutral polarity


def get_the_news(search):
    """Run the main program"""
    template = 'https://news.search.yahoo.com/search?p={}'
    url = template.format(search)
    articles = []
    links = set()
    while True:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        cards = soup.find_all('div', 'NewsArticle')

        # extract articles from page
        for card in cards:
            article = get_article(card)
            link = article[-2]  # The clean_link is now at index -2
            if not link in links:
                links.add(link)
                articles.append(article)

        # find the next page
        try:
            url = soup.find('a', 'next').get('href')
            sleep(1)
        except AttributeError:
            break

    # save article data
    with open('results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Headline', 'Source', 'Posted', 'Description', 'Link', 'Polarity', 'Subjectivity'])
        writer.writerows(articles)

    return articles[:5]


def get_company_info(symbol):
    # Get the stock info for the given symbol
    stock_info = yf.Ticker(symbol)

    # Fetch various company information using the 'info' attribute
    name = stock_info.info.get('longName', 'N/A')
    description = stock_info.info.get('longBusinessSummary', 'N/A')
    #ceo = stock_info.info.get('ceo', 'N/A')
    website = stock_info.info.get('website', 'N/A')
    sector = stock_info.info.get('sector', 'N/A')
    industry = stock_info.info.get('industry', 'N/A')
    market_cap = stock_info.info.get('marketCap', 'N/A')
    trailing_pe = stock_info.info.get('trailingPE', 'N/A')
    full_time_employees = stock_info.info.get('fullTimeEmployees', 'N/A')
    total_revenue = stock_info.info.get('totalRevenue', 'N/A')
    volume = stock_info.info.get('volume', 'N/A')
    current_price = stock_info.info.get('regularMarketPrice', 'N/A')
    target_high_price = stock_info.info.get('targetHighPrice', 'N/A')
    target_low_price = stock_info.info.get('targetLowPrice', 'N/A')
    dividend_yield = stock_info.info.get('dividendYield', 'N/A')
    beta = stock_info.info.get('beta', 'N/A')
    debt_to_equity = stock_info.info.get('debtToEquity', 'N/A')
    company_officers=stock_info.info.get('companyOfficers', 'N/A')

    officers = stock_info.info.get('officers', [])
    officer_names = [officer['name'] for officer in officers] if officers else []

    company_info_dict = {
        'Symbol': symbol,
        'Name': name,
        'Description': description,
        #'CEO': ceo,
        'Website': website,
        'Sector': sector,
        'Industry': industry,
        'Market_Cap': market_cap,
        'Trailing_PE': trailing_pe,
        'Full_Time_Employees': full_time_employees,
        'Total_Revenue': total_revenue,
        'Volume': volume,
        'Current_Price': current_price,
        'Target_High_Price': target_high_price,
        'Target_Low_Price': target_low_price,
        'Dividend_Yield': dividend_yield,
        'Beta': beta,
        'Debt_to_Equity': debt_to_equity,
        'companyOfficers': company_officers,

    }

    return company_info_dict

def predict_future_prices(symbol, duration, prediction_period):
    # Load the trained model from the file
    from keras.models import load_model
    model_fit = load_model('keras_model.h5')

    # Calculate the start and end dates for past 5 years and future prediction
    end_date_past = datetime.now().date()
    start_date_past = end_date_past - timedelta(days=duration * 365)
   
    # Download stock data from Yahoo Finance for past 5 years and future prediction
    past_data = yf.download(symbol, start=start_date_past, end=end_date_past)
    future_dates = pd.date_range(start=end_date_past + timedelta(days=1), periods=prediction_period, freq='D')

    # Get the past actual prices
    past_prices = past_data['Close']

    # Combine past actual prices and future dates
    actual_prices = pd.concat([past_prices, pd.Series(index=future_dates)])

    # Forecast the future prices
    order = (1, 1, 1)  # Set the order of SARIMAX model (p, d, q)
    seasonal_order = (1, 1, 1, 12)  # Set the seasonal order of SARIMAX model (P, D, Q, S)
    model = ARIMA(actual_prices, order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    future_prices = model_fit.predict(start=len(past_prices), end=len(past_prices) + prediction_period - 1)

    return actual_prices, future_dates, future_prices, past_prices


def stock_prediction(request):
    if request.method == 'POST':
        symbol = request.POST.get('symbol')  # Ensure the form field name is correct
        duration = int(request.POST.get('duration'))
        prediction_period = int(request.POST.get('prediction_period'))

        # Retrieve articles with sentiment analysis
        articles = get_the_news(symbol)

        # Predict future prices and retrieve actual prices
        actual_prices, future_dates, future_prices, past_prices = predict_future_prices(symbol, duration, prediction_period)

        y_test, y_predicted = train_model(symbol, duration)

        # Fetch company information directly from the Alpha Vantage API
        company_info = get_company_info(symbol)

        current_price = company_info['Current_Price']
        company_officers = company_info['companyOfficers']

        # Create a Plotly graph
        graph = go.Figure()

        # Add trace for actual prices
        graph.add_trace(go.Scatter(x=actual_prices.index, y=actual_prices.values, mode='lines', name='Actual Price'))

        # Add trace for predicted prices (Future)
        graph.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines', name='Predicted Price (Future)'))

        # Add trace for predicted prices (Past)
        graph.add_trace(go.Scatter(x=actual_prices.index[-len(y_test):], y=y_test.flatten(), mode='lines', name='Predicted Price (Past)'))

        # Add trace for y_predicted
        #graph.add_trace(go.Scatter(x=actual_prices.index[-len(y_test):], y=y_predicted.flatten(), mode='lines', name='y_predicted'))

        article_headlines = []
        article_colors = []
        for article in articles:
            headline, source, _, _, polarity, _ = article
            article_headlines.append(headline)
            color = get_color_from_polarity(polarity)
            article_colors.append(color)

        #graph.add_trace(go.Scatter(x=article_headlines, y=[past_prices.max()] * len(article_headlines),mode='markers', name='Articles', marker=dict(color=article_colors)))
        # Update layout
        graph.update_layout(title='Stock Price Prediction')

        # Convert the graph to HTML
        graph_html = opy.plot(graph, auto_open=False, output_type='div')

        # Prepare the output data
        output_data = {
            'symbol': symbol,
            'duration': duration,
            'graph_html': graph_html,
            'articles': articles,
            'company_info': company_info,
        }

        return render(request, 'prediction.html', {'output_data': output_data})

    return render(request, 'prediction.html')

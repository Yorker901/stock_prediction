import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('multi_stock_model.pkl')

# Streamlit app configuration
st.set_page_config(page_title='Multi-Stock Price Prediction', page_icon=':chart_with_upwards_trend:', layout='wide')
st.title('Multi-Stock Price Prediction Application :chart_with_upwards_trend:')

# Sidebar for inputs
st.sidebar.title('Input Parameters')

# Function to fetch stock tickers with autocomplete
@st.cache
def fetch_stock_tickers():
    return yf.Tickers()

stock_tickers = fetch_stock_tickers()

# Input for stock tickers (multiselect with autocomplete)
selected_tickers = st.sidebar.multiselect('Select Stock Tickers', stock_tickers.symbols, help="Type to search and select")

# Input for prediction date
prediction_date = st.sidebar.date_input('Enter Prediction Date', pd.to_datetime('2024-06-01'))

if st.sidebar.button('Predict'):
    # Initialize a dictionary to store data for each stock
    stock_data_dict = {}

    for ticker in selected_tickers:
        # Fetch historical stock data for each ticker
        stock_data = yf.download(ticker, start='2010-01-01', end=prediction_date)

        if not stock_data.empty:
            # Prepare data for prediction
            stock_data['Date'] = stock_data.index
            stock_data['Date'] = stock_data['Date'].apply(lambda x: x.toordinal())

            X = stock_data[['Date']]
            y = stock_data['Close']

            # Predict future price
            future_date_ordinal = pd.to_datetime(prediction_date).toordinal()
            future_price = model.predict([[future_date_ordinal]])[0]

            # Generate recommendations
            recent_price = y.iloc[-1]
            if future_price > recent_price:
                recommendation = "Buy"
            else:
                recommendation = "Sell"

            # Store data and predictions in the dictionary
            stock_data_dict[ticker] = {
                'data': stock_data,
                'future_price': future_price,
                'recommendation': recommendation
            }

    # Display results for each stock
    for ticker, data in stock_data_dict.items():
        st.write(f'## Predicted Price for {ticker} on {prediction_date}: ${data["future_price"]:.2f}')
        st.write(f'## Investment Recommendation: **{data["recommendation"]}**')

        # Plot historical and predicted prices
        data['data']['Predicted'] = np.nan
        data['data'].loc[data['data'].index[-1], 'Predicted'] = data['future_price']

        plt.figure(figsize=(10, 6))
        plt.plot(data['data']['Close'], label='Historical Prices')
        plt.scatter(data['data'].index[-1], data['future_price'], color='red', label='Predicted Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{ticker} Stock Prices')
        plt.legend()
        st.pyplot(plt)
else:
    st.write("Please select stock tickers and prediction date, then click 'Predict'.")

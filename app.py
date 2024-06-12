import streamlit as st
import pandas as pd
import yfinance as yf
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the trained model
model = joblib.load('reliance_stock_model.pkl')

# Streamlit app configuration
st.set_page_config(page_title='Stock Price Prediction', page_icon=':chart_with_upwards_trend:', layout='wide')
st.title('Stock Price Prediction Application :chart_with_upwards_trend:')

# Sidebar for inputs
st.sidebar.title('Input Parameters')

# Input for stock ticker
stock_ticker = st.sidebar.text_input('Enter Stock Ticker', 'RELIANCE.NS')

# Input for prediction date
prediction_date = st.sidebar.date_input('Enter Prediction Date', pd.to_datetime('2024-06-01'))

if st.sidebar.button('Predict'):
    # Fetch historical stock data
    stock_data = yf.download(stock_ticker, start='2010-01-01', end=prediction_date)
    
    if stock_data.empty:
        st.error("Invalid stock ticker or no data available.")
    else:
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
        
        # Display results
        st.write(f'## Predicted Price for {stock_ticker} on {prediction_date}: ${future_price:.2f}')
        st.write(f'## Investment Recommendation: **{recommendation}**')
        
        # Plot historical and predicted prices
        stock_data['Predicted'] = np.nan
        stock_data.loc[stock_data.index[-1], 'Predicted'] = future_price
        
        plt.figure(figsize=(10, 6))
        plt.plot(stock_data['Close'], label='Historical Prices')
        plt.scatter(stock_data.index[-1], future_price, color='red', label='Predicted Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(f'{stock_ticker} Stock Prices')
        plt.legend()
        st.pyplot(plt)
else:
    st.write("Please input stock ticker and prediction date, then click 'Predict'.")

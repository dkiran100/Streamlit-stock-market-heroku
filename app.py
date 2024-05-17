import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow
from tensorflow import keras
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler

model = load_model('pred.keras')

st.title('Stock Dashboard')

stock = st.text_input('Enter Stock Symnbol')
#start = '2012-01-01'
#end = '2022-12-31'

start_date = st.date_input('Start Date')
end_date = st.date_input('End Date')

data = yf.download(stock, start = start_date , end = end_date)

st.subheader('Stock Data')
st.write(data)

fig = px.line(data, x = data.index, y = data['Adj Close'], title = stock)
st.plotly_chart(fig)

trends, future_trends, pricing_data, news = st.tabs(["Trends", "Future Trends", "Pricing Data", "Top 10 News"])

with pricing_data:
    st.header('Price Movements')
    data2 = data
    data2['% Change'] = data['Adj Close']/data['Adj Close'].shift(1) - 1
    data2.dropna(inplace = True)
    st.write(data2)
    annual_return = data2['% Change'].mean()*252*100
    st.write('Annual return is ', annual_return, '%')
    stdev = np.std(data2['% Change'])*np.sqrt(252)
    st.write('Standard Deviation is ', stdev*100, '%')
    st.write('Risk Adj. Return is ', annual_return/(stdev*100))

from stocknews import StockNews

with news:
    st.header(f'News of {stock}')
    sn = StockNews(stock, save_news = False)
    df_news = sn.read_rss()

    for i in range(10):
        st.subheader(f'News {i+1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment = df_news['sentiment_summary'][i]
        st.write(f'News Sentiment {news_sentiment}')
    
with future_trends:
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

    data = yf.download(stock, START, TODAY)

    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

        
    data_load_state = st.text('Loading data...')
    data = load_data(stock)
    data_load_state.text('Loading data... done!')

    st.subheader('Raw data')
    st.write(data.tail())

    # Plot raw data
    def plot_raw_data():
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
        fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)
        
    plot_raw_data()

    # Predict forecast with Prophet.
    df_train = data[['Date','Close']]
    df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(forecast.tail())
        
    st.write(f'Forecast plot for {n_years} years')
    fig1 = plot_plotly(m, forecast)
    st.plotly_chart(fig1)

    st.write("Forecast components")
    fig2 = m.plot_components(forecast)
    st.write(fig2)



with trends:
    data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)

    st.subheader('Price vs MA50')
    ma_50_days = data.Close.rolling(50).mean()
    fig1 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, 'r')
    plt.plot(data.Close, 'g')
    plt.show()
    st.pyplot(fig1)

    st.subheader('Price vs MA50 vs MA100')
    ma_100_days = data.Close.rolling(100).mean()
    fig2 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, 'r')
    plt.plot(ma_100_days, 'b')
    plt.plot(data.Close, 'g')
    plt.show()
    st.pyplot(fig2)

    st.subheader('Price vs MA100 vs MA200')
    ma_200_days = data.Close.rolling(200).mean()
    fig3 = plt.figure(figsize=(8,6))
    plt.plot(ma_100_days, 'r')
    plt.plot(ma_200_days, 'b')
    plt.plot(data.Close, 'g')
    plt.show()
    st.pyplot(fig3)

    x = []
    y = []

    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i,0])

    x,y = np.array(x), np.array(y)

    predict = model.predict(x)

    scale = 1/scaler.scale_

    predict = predict * scale
    y = y * scale

    st.subheader('Original Price vs Predicted Price')
    fig4 = plt.figure(figsize=(8,6))
    plt.plot(predict, 'r', label='Original Price')
    plt.plot(y, 'g', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()
    st.pyplot(fig4)

import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

#Caching data to reduce wait time
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

#Setting up the plot/graph
def plot_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Open"], name="stock_open"))
    fig.update_traces(line_color='red')
    fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="stock_close"))
    fig.layout.update(xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)


START = "2019-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Price Predicion App")

#Tickers selection box
stock_Tickers = ("TSLA", "AAPL", "META", "KO", "JNJ", "PG", "NVDA", "AMD")
selected_stock = st.selectbox("Select your stock to get a prediction", stock_Tickers)

#Creating the slider
n_years = st.slider("Years of prediction: ", 1, 5)
period_days = n_years * 365

#Interactive interface when loading data
loading_state = st.text("Currently loading...")
data = load_data(selected_stock)
loading_state.text("...Done!")

st.subheader("Market Data")
st.write(data)

st.subheader("Market Summary")
plot_data(data)

#Stock forecast using Prophet
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns = {"Date": "ds", "Close": "y"})

loading_state2 = st.text("Currently loading forecast data...")
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods = period_days)
forecast = m.predict(future)
loading_state2.text("...Done!")

st.subheader(f"Forecast Data({n_years}years)")
st.write(forecast)

st.subheader('Forecast Summary')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.subheader('Forecast Components')
fig2 = m.plot_components(forecast)
st.write(fig2)
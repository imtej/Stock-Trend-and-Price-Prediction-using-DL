import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.metrics import r2_score

# Get the start and end dates 
start = '2010-01-01'
end = '2019-12-31'

# Define the title of the WebApp
st.title('Stock Trend Prediction')

# Take the Stock ticker name for Specific Stocks from the user
user_input = st.text_input('Enter Stock Ticker','AAPL')
start_date = st.text_input('Enter Start Date (YYYY-MM-DD)','2010-01-01')
end_date = st.text_input('Enter End Date(YYYY-MM-DD)','2019-12-31')


# Fetch stock data from Yahoo Finance using the user-defined start and end dates
stock_data = yf.download(user_input, start=start_date, end=end_date)
df = pd.DataFrame(stock_data)


# Describing Data
st.subheader(f'Data from {start_date} to {end_date}')
st.write(df.describe())


# Vizualization
import plotly.graph_objects as go


st.subheader('Closing Price vs Time chart')
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df.Close, name='Closing Price', line=dict(color='green')))

fig.update_layout(
    title='Price Variations',
    xaxis_title='Date-->>',
    yaxis_title='Price-->>',
    width=1300,  # set the width of the figure
    height=900  # set the height of the figure
)

fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')))

st.plotly_chart(fig)



st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100) . mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df.Close, name='Closing Price', line=dict(color='green')))
fig.add_trace(go.Scatter(x=df.index, y=ma100, name='100 Moving Average', line=dict(color='red')))

fig.update_layout(
    title='Price Variations',
    xaxis_title='Date-->>',
    yaxis_title='Price-->>',
    width=1300,  # set the width of the figure
    height=900  # set the height of the figure
)

fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')))

st.plotly_chart(fig)



st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma200 = df.Close.rolling(200) . mean()
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df.Close, name='Closing Price', line=dict(color='green')))
fig.add_trace(go.Scatter(x=df.index, y=ma100, name='100 Moving Average', line=dict(color='red')))
fig.add_trace(go.Scatter(x=df.index, y=ma200, name='200 Moving Average', line=dict(color='blue')))

fig.update_layout(
    title='Price Variations',
    xaxis_title='Date-->>',
    yaxis_title='Price-->>',
    width=1300,  # set the width of the figure
    height=900  # set the height of the figure
)

fig.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')))

st.plotly_chart(fig)


# Splitting Data into Training and Testing

data_training = pd.DataFrame(df["Close"][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df["Close"][int(len(df)*0.70) : int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range =(0,1))

data_training_array = scaler.fit_transform(data_training)

# # Forming Time Series
# x_train = []
# y_train = []

# for i in range(100, data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100:i])
#     y_train.append(data_training_array[i,0])

# x_train, y_train = np.array(x_train), np.array(y_train)   



# # Load my model

import os
import requests
import zipfile
import io

def clone_github_repository(github_repo_url, destination_directory):
    repository_name = github_repo_url.split('/')[-1]
    os.makedirs(destination_directory, exist_ok=True)
    zip_url = f'{github_repo_url}/archive/main.zip'
    response = requests.get(zip_url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_ref:
            zip_ref.extractall(destination_directory)
        print(f'Repository cloned to {destination_directory}/{repository_name}')
    else:
        print(f'Failed to download the repository. Status code: {response.status_code}')

git_url = 'https://github.com/imtej/Stock-Trend-and-Price-Prediction-using-DL'
dest_dir = "files"
clone_github_repository(git_url, dest_dir)


# import os

# # Get the directory of the current script
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Load my model
# model_path = os.path.join(current_dir, 'keras_model.h5')
# model = load_model(model_path)


model = load_model('./files/Stock-Trend-and-Price-Prediction-using-DL-main/keras_model.h5')




# Testing part
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

# forming time series
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)   

# Making Predictions
y_predicted = model.predict(x_test)

scale_factor = 1/scaler.scale_[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

# Final Graph

st.subheader('Prediction vs Original')
fig2 = go.Figure()

fig2.add_trace(go.Scatter(x=df.index[len(df.index)-len(y_test):], y=y_test, name='Test Closing Price',line=dict(color='green')))
fig2.add_trace(go.Scatter(x=df.index[len(df.index)-len(y_test):], y=y_predicted[:,0], name='Predicted Closing Price',line=dict(color='magenta')))

fig2.update_layout(
    title='Price Variations',
    xaxis_title='Date-->>',
    yaxis_title='Price-->>',
    width=1300,  # set the width of the figure
    height=900  # set the height of the figure
)

fig2.update_traces(marker=dict(size=8, line=dict(width=2, color='DarkSlateGrey')))

st.plotly_chart(fig2)




## Calculate R2 score
r2 = r2_score(y_test, y_predicted)

# Display R2 score
st.subheader('Model Evaluation(R2 Score)')
st.write(f'R2 Score: {r2}')





import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from datetime import date
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = date.today()

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

df = data.DataReader(user_input,'yahoo',start,end)

#decribing data
st.subheader('Data from 2010')
st.write(df.describe())

#Visualizations through charts
st.subheader('Closing Price vs Time')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time with 100 days Moving Average')
ma100 = df.Close.rolling(100, min_periods = 50).mean()

fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100)
st.pyplot(fig)

st.subheader('Closing Price vs Time with 100 and 200 days Moving Average')
ma200 = df.Close.rolling(200, min_periods = 100).mean()

fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.legend(['Original Price','100 days Moving Average','200 days Moving Average'])
st.pyplot(fig)

#Splitting Data
from sklearn.model_selection import train_test_split
df_train , df_test = train_test_split(df.Close , test_size = 0.3 , shuffle = False)
df_train = df_train.to_frame()
df_test = df_test.to_frame()

#Scaling down training data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))
train_arr = scaler.fit_transform(df_train)

#Defining training variables
x_train = []
y_train = []

for i in range(100,train_arr.shape[0]):
    x_train.append(train_arr[i-100:i])
    y_train.append(train_arr[i,0])
    
x_train = np.array(x_train)
y_train = np.array(y_train)

#Loading predefined model
model = load_model('keras_model.h5')

#Testing Data

past_100_df = df_train.tail(100)
final_df = past_100_df.append(df_test,ignore_index = True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
    
x_test = np.array(x_test)
y_test = np.array(y_test)

#Predictions
y_pred = model.predict(x_test)
scale_factor = scaler.scale_[0]
y_pred = y_pred*(1/scale_factor)
y_test = y_test*(1/scale_factor)

#Plotting predictions to check accuracy
st.subheader('Predictions vs Original Price')
figf = plt.figure(figsize = (12,6))
plt.plot(y_test , 'b' , label = 'Orginal Price')
plt.plot(y_pred , 'g' , label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(figf)




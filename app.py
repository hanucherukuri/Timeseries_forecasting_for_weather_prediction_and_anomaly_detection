from zipfile import ZipFile
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
# %matplotlib inline
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
import streamlit as st
print(" Library installed")

uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
zip_file = ZipFile(zip_path)
zip_file.extractall()
csv_path = "jena_climate_2009_2016.csv"
df= pd.read_csv(csv_path)

df.rename(columns={"T (degC)": "Temp"}, inplace=True)

train_size = int(len(df) * 0.75)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(train.shape, test.shape)

scaler = StandardScaler()
scaler = scaler.fit(train[['Temp']])
train[['Temp']] = scaler.transform(train[['Temp']])
test[['Temp']] = scaler.transform(test[['Temp']])

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

TIME_STEPS = 288
# reshape to [samples, time_steps, n_features]
X_train, y_train = create_dataset(
  train[['Temp']],
 train[['Temp']],
  TIME_STEPS
)
X_test, y_test = create_dataset(
  test[['Temp']],
  train[['Temp']],
  TIME_STEPS
)
print(X_train.shape, y_train.shape)

#Recalling the saved model
model = keras.models.load_model("../Assignment_2/model_model.h5")

y_pred = model.predict(X_test)
y_train_inv = scaler.inverse_transform(y_train.reshape(1, -1))
y_test_inv = scaler.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = scaler.inverse_transform(y_pred)

def main():
 st.title("Timeseries forecasting for Weather Prediction")
 st.header("This app forecasts Temperature")
 st.write("-------")
 st.title("WEATHER FORECAST")
 st.text_input("Name of the country", ("Germany"))
 st.text_input("Temperature Unit", ("Celsius"))
 st.title("Predicted Temperature")
 agree=st.button("PREDICT")
 if agree:
  st.line_chart(y_pred_inv[:144])
if __name__ == '__main__':
    main()
    


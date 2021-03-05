#Referred from curiousily.com and Keras.io
#Part 2: Timeseries forecasting for Weather prediction
# Loading Libraries

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

"""# Loading Data"""

uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
zip_file = ZipFile(zip_path)
zip_file.extractall()
csv_path = "jena_climate_2009_2016.csv"
df= pd.read_csv(csv_path)

df.rename(columns={"T (degC)": "Temp"}, inplace=True)

plt.rcParams["figure.figsize"] = (20,5)
plt.plot(df["Temp"])
plt.legend();

"""# Spliting Data"""

train_size = int(len(df) * 0.75)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(train.shape, test.shape)

"""# Scaling Data"""

scaler = StandardScaler()
scaler = scaler.fit(train[['Temp']])
train[['Temp']] = scaler.transform(train[['Temp']])
test[['Temp']] = scaler.transform(test[['Temp']])

"""# Creating Dataset"""

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

"""# Model BIDirectional RNN"""

model = keras.Sequential()
model.add(
  keras.layers.Bidirectional(
    keras.layers.LSTM(
      units=128,
      input_shape=(X_train.shape[1], X_train.shape[2])
    )
  )
)
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.25,
    shuffle=False
)

"""# Ploting Loss"""

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend();

y_pred = model.predict(X_test)

"""# Inverse Transformation"""

y_train_inv = scaler.inverse_transform(y_train.reshape(1, -1))
y_test_inv = scaler.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = scaler.inverse_transform(y_pred)

plt.plot(np.arange(0, len(y_train)), y_train_inv.flatten(), 'g', label="history")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_inv.flatten(), marker='.', label="true")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Temp')
plt.xlabel('Time Step')
plt.legend()
plt.show();

plt.plot(y_test_inv.flatten(), marker='.', label="true")
plt.plot(y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Temp')
plt.xlabel('Time Step')
plt.legend()
plt.show();

"""# Model LSTM"""

model = keras.Sequential()
model.add(
    keras.layers.LSTM(
      units=256,
      input_shape=(X_train.shape[1], X_train.shape[2])
    )
  )
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=256,
    validation_split=0.25,
    shuffle=False
)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend();

y_pred = model.predict(X_test)

y_train_inv = scaler.inverse_transform(y_train.reshape(1, -1))
y_test_inv = scaler.inverse_transform(y_test.reshape(1, -1))
y_pred_inv = scaler.inverse_transform(y_pred)

plt.plot(np.arange(0, len(y_train)), y_train_inv.flatten(), 'g', label="history")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_test_inv.flatten(), marker='.', label="true")
plt.plot(np.arange(len(y_train), len(y_train) + len(y_test)), y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Temp')
plt.xlabel('Time Step')
plt.legend()
plt.show();

plt.plot(y_test_inv.flatten(), marker='.', label="true")
plt.plot(y_pred_inv.flatten(), 'r', label="prediction")
plt.ylabel('Temp')
plt.xlabel('Time Step')
plt.legend()
plt.show();


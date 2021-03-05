#Referred from curiousily.com and Keras.io
#Part 2: Timeseries forecasting for anomaly detection

# Importing Libraries

# Commented out IPython magic to ensure Python compatibility.
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

"""# Loading Dataset"""

uri = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
zip_path = keras.utils.get_file(origin=uri, fname="jena_climate_2009_2016.csv.zip")
zip_file = ZipFile(zip_path)
zip_file.extractall()
csv_path = "jena_climate_2009_2016.csv"
df= pd.read_csv(csv_path)

df.shape

df.corr()

df.rename(columns={"T (degC)": "Temp"}, inplace=True)

plt.rcParams["figure.figsize"] = (20,5)
plt.plot(df["Temp"])
plt.legend();

df.Temp.max()

df.Temp.min()

df.Temp.mean()

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
print(X_train.shape)

"""# Model"""

model = keras.Sequential()
model.add(keras.layers.LSTM(
    units=64,
    input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
model.add(keras.layers.LSTM(units=64, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))
model.add(
  keras.layers.TimeDistributed(
    keras.layers.Dense(units=X_train.shape[2])
  )
)
model.compile(loss='mae', optimizer='adam')

model.summary()

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=256,
    validation_split=0.25,
    shuffle=False
)

"""# Predictions and Loss"""

X_train_pred = model.predict(X_train)
train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend();

sns.distplot(train_mae_loss, bins=50, kde=True);

X_test_pred = model.predict(X_test)

test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

sns.distplot(test_mae_loss, bins=50, kde=True);

"""# Setting Threshold"""

THRESHOLD = 1.1

test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
test_score_df['loss'] = test_mae_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
test_score_df['Temp'] = test[TIME_STEPS:].Temp

plt.plot(test_score_df.index, test_score_df.loss, label='loss')
plt.plot(test_score_df.index, test_score_df.threshold, label='threshold')
plt.xticks(rotation=25)
plt.legend();

"""# Anomalies"""

anomalies = test_score_df[test_score_df.anomaly == True]
anomalies.head()

anomalies.shape

plt.plot(
  test[TIME_STEPS:].index, 
  scaler.inverse_transform(test[TIME_STEPS:].Temp), 
  label='Temp'
);

sns.scatterplot(
  anomalies.index,
  scaler.inverse_transform(anomalies.Temp),
  color=sns.color_palette()[3],
  s=52,
  label='anomaly'
)
plt.xticks(rotation=25)
plt.legend();


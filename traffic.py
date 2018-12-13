import pandas as pd
import numpy as np
import os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import concatenate, SimpleRNN, Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector, CuDNNLSTM, RNN
from keras.models import Sequential, Input, Model
from keras.optimizers import RMSprop, Adamax, SGD
from sklearn import preprocessing
from sklearn.utils import shuffle

VALIDATION_SPLIT = 0.9


# traindf = df[(df['Timestamp'] > '2010/1/3 00:30:00') & (df['Timestamp'] < '2010/1/4 11:50:00')]
def datapreproccess(csv):
    df = pd.read_csv(csv)
    df = df[['Timestamp', 'Average Speed', 'Total Flow', 'Average Occupancy', 'Lane 1 Flow', 'Lane 1 Average Occupancy', 'Lane 1 Average Speed', 'Lane 2 Flow', 'Lane 2 Average Occupancy'
        , 'Lane 2 Average Speed', 'Lane 3 Flow', 'Lane 3 Average Occupancy', 'Lane 3 Average Speed', 'Lane 4 Flow', 'Lane 4 Average Occupancy', 'Lane 4 Average Speed']]
    df['Average  Speed_t2'] = df['Average Speed'].shift(-2)
    df['Average  Speed_t2'] = df['Average  Speed_t2'].fillna(df['Average  Speed_t2'].mean())
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['date'] = df['Timestamp'].dt.day
    df['day'] = df['Timestamp'].dt.dayofweek
    df['hour'] = df['Timestamp'].dt.hour
    df['minute'] = df['Timestamp'].dt.minute
    colist = ['Average  Speed_t2', 'Average Speed', 'Total Flow', 'Average Occupancy', 'Lane 1 Flow', 'Lane 1 Average Occupancy', 'Lane 1 Average Speed', 'Lane 2 Flow', 'Lane 2 Average Occupancy'
        , 'Lane 2 Average Speed', 'Lane 3 Flow', 'Lane 3 Average Occupancy', 'Lane 3 Average Speed', 'Lane 4 Flow', 'Lane 4 Average Occupancy', 'Lane 4 Average Speed']
    # colist = ['Average  Speed_t2', 'Average Speed']
    df = df.reindex(columns=colist)
    df_norm = normalize(df)
    feature, label= buildTrain(df, df_norm, 24, 1)

    # feature = data[:, 1:]
    return label, feature

def normalize(train):
    train = train.drop(["Average  Speed_t2"], axis=1)
    train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    return train_norm

def buildTrain(train, train_norm, past=12, future=1):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-future-past):
        X_train.append(np.array(train_norm.iloc[i:i+past]))
        Y_train.append(np.array(train.iloc[i+past:i+past+future]["Average  Speed_t2"]))
    return np.array(X_train), np.array(Y_train)
def shuffle(X,Y):
    np.random.seed(10)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]

def splitData(X,Y,rate):
    # X_train = X[int(X.shape[0]*rate):]
    # Y_train = Y[int(Y.shape[0]*rate):]
    # X_val = X[:int(X.shape[0]*rate)]
    # Y_val = Y[:int(Y.shape[0]*rate)]
    X_train = X
    Y_train = Y
    X_val = X
    Y_val = Y
    return X_train, Y_train, X_val, Y_val


label, feature = datapreproccess('traindata.csv')
feature, label = shuffle(feature, label)
x_train, y_train, x_val, y_val = splitData(feature, label, VALIDATION_SPLIT)
# y_train = y_train[:, np.newaxis]
# y_val = y_val[:, np.newaxis]
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)


model = Sequential()
model.add(LSTM(1000, input_length=x_train.shape[1], input_dim=x_train.shape[2]))
model.add(Dense(500, activation='relu'))
model.add(Dense(1, activation='relu'))

optAMX = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
optSGD = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
optRMS = RMSprop(lr=0.0005, rho=0.9, epsilon=None, decay=0.0)
model.summary()
model.compile(loss='mse', optimizer=optRMS)
checkpoint = ModelCheckpoint('best.model', monitor='val_loss', verbose=1, 
    save_best_only=True, mode='min')
# LearningRateScheduler(schedule, verbose=0)
callbacks_list = [checkpoint ]
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10000, batch_size=24,
    callbacks=callbacks_list)
import pandas as pd
import numpy as np
import os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Dense, Dropout, Activation, LSTM, CuDNNLSTM, CuDNNGRU, ReLU, Conv1D, MaxPooling1D
from keras.models import Sequential, Input, Model, load_model
from keras.optimizers import RMSprop, Adamax, SGD
from sklearn import preprocessing
from sklearn.utils import shuffle

VALIDATION_SPLIT = 0.9


# traindf = df[(df['Timestamp'] > '2010/1/3 00:30:00') & (df['Timestamp'] < '2010/1/4 11:50:00')]
def datapreproccess(csv):
    df = pd.read_csv(csv)
    df = df[['Timestamp', 'Average Speed', 'Total Flow', 'Average Occupancy', 'Lane 1 Flow', 'Lane 1 Average Occupancy', 'Lane 1 Average Speed', 'Lane 2 Flow', 'Lane 2 Average Occupancy'
        , 'Lane 2 Average Speed', 'Lane 3 Flow', 'Lane 3 Average Occupancy', 'Lane 3 Average Speed', 'Lane 4 Flow', 'Lane 4 Average Occupancy', 'Lane 4 Average Speed']]
    df['Average  Speed_t2'] = df['Average Speed'].shift(-6)
    df['Average  Speed_t2'] = df['Average  Speed_t2'].fillna(df['Average  Speed_t2'].mean())
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['date'] = df['Timestamp'].dt.day
    df['day'] = df['Timestamp'].dt.dayofweek
    df['hour'] = df['Timestamp'].dt.hour
    df['minute'] = df['Timestamp'].dt.minute
    colist = ['Average  Speed_t2', 
        'Average Speed', 'Total Flow', 'Average Occupancy', 
        'Lane 1 Flow', 'Lane 1 Average Occupancy', 'Lane 1 Average Speed',
        'Lane 2 Flow', 'Lane 2 Average Occupancy', 'Lane 2 Average Speed',
        'Lane 3 Flow', 'Lane 3 Average Occupancy', 'Lane 3 Average Speed',
        'Lane 4 Flow', 'Lane 4 Average Occupancy', 'Lane 4 Average Speed']
    # colist = ['Average  Speed_t2', 'Average  Speed_t4', 'Average  Speed_t3', 'Average  Speed_t1', 'Average Speed',
    # 'Lane 1 Average Speed', 'Lane 2 Average Speed', 'Lane 3 Average Speed', 'Lane 4 Average Speed']
    # colist = ['Average  Speed_t2', 'Average Speed', 'Total Flow', 'Average Occupancy', 'day', 'hour', 'minute']
    # colist = ['Average  Speed_t2', 'Average Speed', 'Total Flow', 'Average Occupancy']
    df = df.reindex(columns=colist)
    df_norm = normalize(df)
    feature, label= buildTrain(df, df_norm, 1000, 1)

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


label, feature = datapreproccess('5weeks_traffic.csv')
# feature, label = shuffle(feature, label)
x_train, y_train, x_val, y_val = splitData(feature, label, VALIDATION_SPLIT)
# y_train = y_train[:, np.newaxis]
# y_val = y_val[:, np.newaxis]
print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)


model = Sequential()
# model.add(Dropout(0.2))
# model.add(CuDNNLSTM(20, input_shape=(100, 8), return_sequences=True))
# model.add(Activation('relu'))
# model.add(CuDNNLSTM(15, return_sequences=True))
# model.add(Activation('relu'))
# model.add(CuDNNLSTM(10, return_sequences=False))
# model.add(Activation('tanh'))
# model.add(RepeatVector(1))
# model.add(CuDNNLSTM(10, return_sequences=True))
# model.add(Activation('tanh'))
# model.add(CuDNNLSTM(15, return_sequences=True))
# model.add(Activation('relu'))
# model.add(CuDNNLSTM(20, return_sequences=True))
# model.add(Activation('relu'))
# model.add(TimeDistributed(Dense(10, activation='relu')))
# model.add(TimeDistributed(Dense(1)))

model.add(Conv1D(filters=64, kernel_size=2, padding='valid', activation='relu', input_shape=(1000,15)))
model.add(Conv1D(filters=64, kernel_size=2, padding='valid', activation='relu'))
model.add(MaxPooling1D(pool_size=3))
model.add(CuDNNGRU(200))
model.add(Activation('relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='relu'))

# model.add(CuDNNGRU(10, input_shape=(3000, 6), return_sequences=False))
# model.add(Activation('relu'))
# # model.add(Dropout(0.2))
# model.add(Dense(100))
# # model.add(ReLU(max_value=50))
# model.add(Activation('relu'))
# model.add(Dense(1))
# # model.add(ReLU(max_value=74.4, threshold=12.9))
# model.add(Activation('relu'))

optAMX = Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
optSGD = SGD(lr=0.001, momentum=0.0, decay=0.0, nesterov=False)
optRMS = RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0)
checkpoint = ModelCheckpoint('best.model', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
model.compile(loss='mse', optimizer=optRMS)
# LearningRateScheduler(schedule, verbose=0)
callbacks_list = [checkpoint]
model.summary()
model.load_weights('best.model')
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=3000, batch_size=18,
    callbacks=callbacks_list)

model.load_weights('best.model')
array = model.predict(x=[x_train])
ans = [] 
for row in array: 
    ans.append(row[0])
# print(ans)

i = 0
with open('ans1.csv', 'w') as f:
    for a in array:
        f.write(str(i) + ',' + str(round(a[0], 1)) + '\n')
        i += 1
print(i)
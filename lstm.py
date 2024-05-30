from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Dropout
from keras.layers import Dense, Bidirectional
from keras.preprocessing.sequence import TimeseriesGenerator
import quantlib.general_utils as gu
import pandas as pd
import numpy as np
import math
import plotly.graph_objects as go

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def split_sequence_dat(sequence, d_seq, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix]+ d_seq[end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
n_steps = 3
n_features = 1

def get_vanilla_lstm(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def fit_vanilla_lstm(model, raw_seq, n_steps, n_features):
    # define input sequence 
    # choose a number of time steps
    
    # split into samples
    X, y = split_sequence(raw_seq, n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model

    # fit model
    model.fit(X, y, epochs=200, verbose=0)
    return model

def fit_vanilla_lstm_dat(model, raw_seq, d_seq, n_steps, n_features):
    # define input sequence 
    # choose a number of time steps
    
    # split into samples
    X, y = split_sequence_dat(raw_seq, d_seq,  n_steps)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model

    # fit model
    model.fit(X, y, epochs=200, verbose=0)
    return model

x_input = np.array([70, 80, 90])

def predict_vanilla_lstm(model, x_input, n_steps, n_features):
    # demonstrate prediction
    # x_input = array([70, 80, 90])
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    # print(yhat)
    return yhat

# rmodel = vanilla_lstm(raw_seq, n_steps, n_features)
# vanilla_predict(rmodel, x_input, n_steps, n_features)


def load_local_data():
    df, instruments = gu.load_file("./Data/data.obj")
    # df = du.extend_dataframe(instruments, df)
    #print(instruments)
    return df, instruments

df, instruments = load_local_data()

# print(df.head())
# print(df.tail())
def predict_inst_vol(df, inst, goval):
    columns = [inst+' open', inst+' high', inst+' low', inst+' close', inst+' volume']
    #columns = ['AAL open', 'AAL high', 'AAL low', 'AAL close', 'AAL volume']


    inst_df = df[columns]

    #print(inst_df.head())
    inst_volume = inst_df[inst+' '+goval].tolist()

    data = np.array(inst_volume) 
    nan_indices = np.isnan(data) 
    # print("NANAs", nan_indices)

    inst_size = len(inst_volume)

    inst_train = inst_volume[:-5]

    n_features = 20
    vmodel = get_vanilla_lstm(n_features, 1)
    for i in range(5):
        vmodel = fit_vanilla_lstm(vmodel, inst_train, n_features, 1)
        x_input = np.array(inst_train[-n_features:])
        
        y = predict_vanilla_lstm(vmodel, x_input, n_features, 1)
        print("y -- ", y, y[0][0])
        if(y[0][0] != None and not math.isnan(y[0][0])):
            # print("y -- ", y, int(y[0][0]))
            inst_train.append(float(y[0][0]))

    print(len(inst_volume), inst_volume[-12:])
    print(len(inst_train), inst_train[-10:])

# predict_inst_vol(df, 'CRWD', 'close')

def predict_inst_replay(df, inst, goval):
    columns = [inst+' open', inst+' high', inst+' low', inst+' close', inst+' volume']
    #columns = ['AAL open', 'AAL high', 'AAL low', 'AAL close', 'AAL volume']


    inst_df = df[columns]

    #print(inst_df.head())
    inst_volume = inst_df[inst+' '+goval].tolist()

    data = np.array(inst_volume) 
    nan_indices = np.isnan(data) 
    # print("NANAs", nan_indices)

    inst_size = len(inst_volume)

    inst_train = inst_volume[:-5]

    n_features = 5
    vmodel = get_vanilla_lstm(n_features, 1)
    vmodel = fit_vanilla_lstm(vmodel, inst_train, n_features, 1)
    for i in range(5):
        # vmodel = fit_vanilla_lstm(vmodel, inst_train, n_features, 1)
        x_input = np.array(inst_train[-n_features:])
        
        y = predict_vanilla_lstm(vmodel, x_input, n_features, 1)
        print("y -- ", y, y[0][0], x_input)
        if(y[0][0] != None and not math.isnan(y[0][0])):
            # print("y -- ", y, int(y[0][0]))
            inst_train.append(float(y[0][0]))

    print(len(inst_volume), inst_volume[-12:])
    print(len(inst_train), inst_train[-10:])

# predict_inst_replay(df, 'CRWD', 'close')

def predict_inst_vol_dat(df, inst):
    df_date = df.index.values
    # df1 = pd.DataFrame(index=df.index)
    # df2 = df1.tz_localize(None)
    # print(df_date[0])
    # print(df_date[0].weekday())

    def codr(d):
        code = [0, 0, 0, 0, 0]
        code[d.weekday()-1]=1
        return code
    # x = lambda d : code[d.weekday()-1]=1
    codays = [codr(d) for d in df_date]

    columns = [inst+' open', inst+' high', inst+' low', inst+' close', inst+' volume']
    #columns = ['AAL open', 'AAL high', 'AAL low', 'AAL close', 'AAL volume']


    inst_df = df[columns]

    #print(inst_df.head())
    inst_volume = inst_df[inst+' volume'].tolist()

    data = np.array(inst_volume) 
    nan_indices = np.isnan(data) 
    # print("NANAs", nan_indices)

    inst_size = len(inst_volume)

    inst_train = inst_volume[:-5]

    vmodel = get_vanilla_lstm(25, 1)

    for i in range(5):
        vmodel = fit_vanilla_lstm_dat(vmodel, inst_train, codays, 20, 1)
        x_input = np.array(inst_train[-20:] + codays[i-5])
        
        y = predict_vanilla_lstm(vmodel, x_input, 25, 1)
        print("y -- ", y, y[0][0])
        if(y[0][0] != None and not math.isnan(y[0][0])):
            print("y -- ", y, float(y[0][0]))
            inst_train.append(float(y[0][0]))

    print(len(inst_volume), inst_volume[-10:])
    print(len(inst_train), inst_train[-10:])

# predict_inst_vol_dat(df, 'AAL')
    
# df_date = df.index.values

# print(df_date[0])
# print(df_date[0].weekday())

# def codr(d):
#     code = [0, 0, 0, 0, 0]
#     code[d.weekday()-1]=1
#     return code

# codays = [codr(d) for d in df_date]
# print(codays[1:4])
# print(len(codays))
# print(len(df_date.tolist()))


def tg_vanilla(df1, inst, goval):
    cols = [inst+' open', inst+' high', inst+' low', inst+' volume']
    
    df = df1[[inst+' open', inst+' high', inst+' low', inst+' close', inst+' volume', inst+' % active']]
    df = df[df[inst+' % active']]
    df = df[1:]
    df['Date'] = pd.to_datetime(df.index.values)
    df = df.set_axis(df['Date'])
    df = df.drop(columns=cols)
    print(df.head())

    close_data = df[inst+' close'].values
    close_data = close_data.reshape((-1,1))

    split_percent = 0.80
    split = int(split_percent*len(close_data))

    close_train = close_data[:split]
    close_test = close_data[split:]

    date_train = df['Date'][:split]
    date_test = df['Date'][split:]

    print(len(close_train))
    print(len(close_test))

    look_back = 15

    train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)     
    test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

    model = Sequential()
    model.add(
        LSTM(10,
            activation='relu', return_sequences=True,
            input_shape=(look_back,1))
    )
    model.add(Dropout(0.2))
    model.add(
        LSTM(10,
            activation='relu',
            return_sequences=True)
    )
    model.add(Dropout(0.2))
    model.add(
        LSTM(10,
            activation='relu',
            return_sequences=True)
    )
    model.add(Dropout(0.2))
    model.add(
        LSTM(10,
            activation='relu')
    )
    model.add(Dropout(0.2))    
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    num_epochs = 25
    model.fit_generator(train_generator, epochs=num_epochs, verbose=1)

    prediction = model.predict_generator(test_generator)

    close_train = close_train.reshape((-1))
    close_test = close_test.reshape((-1))
    prediction = prediction.reshape((-1))

    trace1 = go.Scatter(
        x = date_train,
        y = close_train,
        mode = 'lines',
        name = 'Data'
    )
    trace2 = go.Scatter(
        x = date_test,
        y = prediction,
        mode = 'lines',
        name = 'Prediction'
    )
    trace3 = go.Scatter(
        x = date_test,
        y = close_test,
        mode='lines',
        name = 'Ground Truth'
    )
    layout = go.Layout(
        title = "Google Stock",
        xaxis = {'title' : "Date"},
        yaxis = {'title' : "Close"}
    )
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    fig.show()


# tg_vanilla(df, 'CRWD', 'close')
    


def tg_bidirectional(df1, inst, goval):
    cols = [inst+' open', inst+' high', inst+' low', inst+' volume']
    
    df = df1[[inst+' open', inst+' high', inst+' low', inst+' close', inst+' volume', inst+' % active']]
    df = df[df[inst+' % active']]
    df = df[1:]
    df['Date'] = pd.to_datetime(df.index.values)
    df = df.set_axis(df['Date'])
    df = df.drop(columns=cols)

    close_data = df[inst+' close'].values
    close_data = close_data.reshape((-1,1))

    split_percent = 0.80
    split = int(split_percent*len(close_data))

    look_back = 13

    close_train = close_data[:split]
    close_test = close_data[split-look_back:]

    date_train = df['Date'][:split]
    date_test = df['Date'][split:]

    print(len(close_train))
    print(len(close_test))

    train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)     
    test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=10)

    model = Sequential()
    model.add(Bidirectional(LSTM(10, activation='relu', return_sequences=False), input_shape=(look_back,1)))
    # model.add(Dropout(0.02))
    # model.add(Bidirectional(LSTM(10, return_sequences=True)))
    # model.add(Dropout(0.02))    
    # model.add(Bidirectional(LSTM(10, activation='relu', return_sequences=True)))
    # model.add(Dropout(0.02))
    # model.add(Bidirectional(LSTM(10, return_sequences=True)))
    # model.add(Dropout(0.02))       
    # model.add(Bidirectional(LSTM(10, activation='relu')))
    # model.add(Dropout(0.02))  
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    num_epochs = 210
    model.fit_generator(train_generator, epochs=num_epochs, verbose=1)

    prediction = model.predict_generator(test_generator)

    close_train = close_train.reshape((-1))
    close_test = close_test.reshape((-1))
    prediction = prediction.reshape((-1))

    trace1 = go.Scatter(
        x = date_train,
        y = close_train,
        mode = 'lines',
        name = 'Data'
    )
    print(date_test[-5:])
    print(len(date_test))
    print(len(prediction))
    print(len(close_test))
    # date_test1 = date_test[-len(prediction):]
    trace2 = go.Scatter(
        x = date_test,
        y = prediction,
        mode = 'lines',
        name = 'Prediction'
    )
    trace3 = go.Scatter(
        x = date_test,
        y = close_test[look_back:],
        mode='lines',
        name = 'Ground Truth'
    )
    # layout = go.Layout(
    #     title = inst+" Stock",
    #     xaxis = {'title' : "Date"},
    #     yaxis = {'title' : "Close"}
    # )
    # fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    # fig.show()

    close_data = close_data.reshape((-1))

    def predict(num_prediction, model):
        prediction_list = close_data[-look_back:]
        
        for _ in range(num_prediction):
            x = prediction_list[-look_back:]
            x = x.reshape((1, look_back, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[look_back:]
            
        return prediction_list
        
    def predict_dates(num_prediction):
        last_date = df['Date'].values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
        return prediction_dates[1:]

    num_prediction = look_back - 7
    forecast = predict(num_prediction, model)
    forecast_dates = predict_dates(num_prediction)

    trace4 = go.Scatter(
        x = forecast_dates,
        y = forecast,
        mode='lines',
        name = 'Future Values'
    )
    layout = go.Layout(
        title = inst+" Stock",
        xaxis = {'title' : "Date"},
        yaxis = {'title' : "Close"}
    )
    fig = go.Figure(data=[trace1, trace2, trace3, trace4], layout=layout)
    fig.show()
tg_bidirectional(df, 'SAP', 'close')
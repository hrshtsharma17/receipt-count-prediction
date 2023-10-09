import sys
sys.path.append('/Users/harshitsharma/CoStacks/Fetch')

import os
import pandas as pd
from datetime import datetime
from model.data_processing import prepare_data
from model.evaluation import model_eval

import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

def RCLSTM(n_steps, n_features):
    model = Sequential()
    model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(Dropout(0.3))  # Example dropout layer with a dropout rate of 0.3
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    
    return model

if __name__=="__main__":
    timeseries_data = pd.read_csv('data/data_daily.csv')

    n_steps = 120 # choose a number of time steps
    X, y = prepare_data(timeseries_data["Receipt_Count"], n_steps)
    n_features = 1 # we can add more features later such as is_weekend etc

    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    size = len(X)
    train_size = int(size*0.9)

    # 10% of future values to test dataset
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_folder = f'model/artifacts_{current_datetime}'
    os.makedirs(output_folder, exist_ok=True)

    model = RCLSTM(n_steps, n_features)
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))

    # Define early stopping and learning rate reduction callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
    model_checkpoint = ModelCheckpoint(os.path.join(output_folder, 'best_model.h5'), 
                                    monitor='val_loss', save_best_only=True)

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, verbose=1, validation_split=0.1,
                        callbacks=[early_stopping, reduce_lr, model_checkpoint])

    model_eval(X_test, y_test, model, output_folder)
    print("Results saved to %s" % output_folder)
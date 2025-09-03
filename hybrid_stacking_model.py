import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

class HybridStackingModel:
    def __init__(self, lstm_epochs=10, lstm_batch_size=32):
        self.lr = LinearRegression()
        self.rf = RandomForestRegressor(n_estimators=100, random_state=42)
        self.xgb = xgb.XGBRegressor(n_estimators=100, random_state=42)
        self.lstm_epochs = lstm_epochs
        self.lstm_batch_size = lstm_batch_size
        self.lstm = None
        self.meta_model = LinearRegression()
        self.feature_names = None

    def fit_lstm(self, X, y):
        # Reshape for LSTM: [samples, timesteps, features]
        X_lstm = np.expand_dims(X.values.astype(np.float32), axis=1)
        y_lstm = y.values.astype(np.float32)
        model = Sequential()
        model.add(LSTM(32, input_shape=(X_lstm.shape[1], X_lstm.shape[2]), activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer=Adam(), loss='mse')
        model.fit(X_lstm, y_lstm, epochs=self.lstm_epochs, batch_size=self.lstm_batch_size, verbose=0)
        self.lstm = model

    def predict_lstm(self, X):
        X_lstm = np.expand_dims(X.values.astype(np.float32), axis=1)
        return self.lstm.predict(X_lstm).flatten()

    def fit(self, X, y):
        self.feature_names = X.columns.tolist()
        # Fit base models
        self.lr.fit(X, y)
        self.rf.fit(X, y)
        self.xgb.fit(X, y)
        self.fit_lstm(X, y)

    def predict_base(self, X):
        preds = {
            'lr': self.lr.predict(X),
            'rf': self.rf.predict(X),
            'xgb': self.xgb.predict(X),
            'lstm': self.predict_lstm(X)
        }
        return pd.DataFrame(preds, index=X.index)

    def fit_meta(self, X, y):
        base_preds = self.predict_base(X)
        self.meta_model.fit(base_preds, y)

    def predict(self, X):
        base_preds = self.predict_base(X)
        return self.meta_model.predict(base_preds)

    def fit_stacking(self, X_train, y_train, X_valid, y_valid):
        # Fit base models on train
        self.fit(X_train, y_train)
        # Fit meta model on validation set base predictions
        self.fit_meta(X_valid, y_valid)

    def score(self, X, y):
        preds = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        return rmse

# Example usage:
# df = ... # your processed dataframe
# features = ... # your feature columns
# X = df[features]
# y = df['Sales']
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
# model = HybridStackingModel(lstm_epochs=20)
# model.fit_stacking(X_train, y_train, X_valid, y_valid)
# print('Validation RMSE:', model.score(X_valid, y_valid))
# preds = model.predict(X_valid)

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# データの読み込み
# train_model.pyで使うCSVファイル名を指定
# このファイルはget_data.pyで作成したものを使います
file_name = "7203.T_stock_data.csv"
file_path = f"data/{file_name}"

data = pd.read_csv(file_path, header=1)
# 列名を6つに修正
data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
data = data.dropna()  # 欠損値がある行を削除
close_prices = data['Close'].values.reshape(-1, 1)

# データの正規化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# 学習データとテストデータの準備
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[0:train_size, :]
test_data = scaled_data[train_size:len(scaled_data), :]

def create_dataset(dataset, look_back=60):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

look_back = 60
X_train, y_train = create_dataset(train_data, look_back)
X_test, y_test = create_dataset(test_data, look_back)

# LSTMの入力形式にデータを整形
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# LSTMモデルの構築
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# モデルのコンパイルと学習
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=1, epochs=1)

# 学習済みモデルの保存
if not os.path.exists("models"):
    os.makedirs("models")
model.save("models/stock_predictor_model.h5")

print("モデルの学習が完了し、'models'フォルダに保存しました。")
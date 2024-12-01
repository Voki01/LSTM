from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

# Загрузка данных
data = pd.read_csv("radiator_data.csv")
X_train = data[["Temperature", "Price", "CompetitorPrice", "Discount"]].values
y_train = data[["AluminumRadiators", "CopperRadiators"]].values

# Нормализация данных
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train = scaler_x.fit_transform(X_train)
y_train = scaler_y.fit_transform(y_train)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))

# Проверяем, существует ли уже сохранённая модель
model_path = "lstm_model.keras"
if os.path.exists(model_path):
    print("Загрузка обученной модели...")
    model = load_model(model_path)
else:
    print("Обучение модели...")
    # Создаём модель
    model = Sequential([
        LSTM(100, activation='relu', return_sequences=True, input_shape=(1, 4)),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse')

    # Обучаем модель
    model.fit(X_train, y_train, epochs=500, verbose=1)

    # Сохраняем модель
    model.save(model_path)
    print(f"Модель сохранена в {model_path}")

# Flask API для прогнозов
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    inputs = np.array([
        data['averageTemperature'],
        data['price'],
        data['competitorPrice'],
        data['discount']
    ])
    inputs = scaler_x.transform([inputs])
    inputs = inputs.reshape((1, 1, 4))
    prediction = model.predict(inputs, verbose=0)
    prediction = scaler_y.inverse_transform(prediction)

    return jsonify({
        "aluminum": float(prediction[0][0]),
        "copper": float(prediction[0][1])
    })

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import json
import os

app = Flask(__name__)

# Пути к файлам
model_path = "lstm_model.keras"
history_path = "training_history.json"

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
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Указана метрика 'mae'

    # Обучаем модель
    history = model.fit(X_train, y_train, epochs=50, verbose=1)

    # Сохраняем историю обучения
    training_history = {
        "epochs": list(range(1, len(history.history["loss"]) + 1)),
        "loss": history.history["loss"],
        "accuracy": [1 - mae for mae in history.history["mae"]]  # "Точность" на основе MAE
    }
    with open(history_path, "w") as f:
        json.dump(training_history, f)

    # Сохраняем модель
    model.save(model_path)
    print(f"Модель сохранена в {model_path}")

# Проверяем наличие истории обучения
if not os.path.exists(history_path):
    print("История обучения отсутствует. Пересчёт метрик...")
    history = model.fit(X_train, y_train, epochs=10, verbose=1)
    training_history = {
        "epochs": list(range(1, len(history.history["loss"]) + 1)),
        "loss": history.history["loss"],
        "accuracy": [1 - mae for mae in history.history["mae"]]
    }
    with open(history_path, "w") as f:
        json.dump(training_history, f)

# Метрики модели
@app.route('/metrics', methods=['GET'])
def get_metrics():
    # Рассчитываем метрики на обучающем наборе
    results = model.evaluate(X_train, y_train, verbose=0)
    if isinstance(results, list):  # Если несколько метрик
        loss = results[0]
        mae = results[1]
    else:  # Если одна метрика
        loss = results
        mae = 0  # Указываем значение по умолчанию

    accuracy = 1 - mae  # "Точность" в диапазоне от 0 до 1

    return jsonify({
        "MSE": loss,
        "Accuracy": accuracy
    })

# Прогресс обучения
@app.route('/training-progress', methods=['GET'])
def get_training_progress():
    try:
        with open(history_path, "r") as f:
            history = json.load(f)
        return jsonify({
            "epochs": history["epochs"],
            "loss": history["loss"],
            "accuracy": history["accuracy"]
        })
    except FileNotFoundError:
        return jsonify({
            "epochs": [],
            "loss": [],
            "accuracy": []
        })

# Архитектура сети
@app.route('/architecture', methods=['GET'])
def get_architecture():
    architecture = [
        {"Type": "LSTM", "Output Shape": "(None, 100)", "Units": 100},
        {"Type": "LSTM", "Output Shape": "(None, 50)", "Units": 50},
        {"Type": "Dense", "Output Shape": "(None, 2)", "Units": 2}
    ]
    return jsonify(architecture)

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

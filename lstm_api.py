from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.saving import register_keras_serializable
import tensorflow as tf
import json
import os

app = Flask(__name__)

# Пути к файлам
model_path = "lstm_model.h5"
history_path = "training_history.json"

# Регистрируем метрику MSE
@register_keras_serializable()
def custom_mse(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred)

# Загрузка данных
data = pd.read_csv("radiator_data.csv")
X = data[["Temperature", "Price", "CompetitorPrice", "Discount"]].values
y = data[["AluminumRadiators", "CopperRadiators"]].values

# Нормализация данных
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y)

# Разделение на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Изменяем форму данных для LSTM
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Проверяем, существует ли уже сохранённая модель
if os.path.exists(model_path):
    print("Загрузка обученной модели...")
    model = load_model(model_path, custom_objects={'mse': custom_mse})
else:
    print("Обучение модели...")
    model = Sequential([
        LSTM(100, activation='relu', return_sequences=True, input_shape=(1, 4)),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(2)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    training_history = {"epochs": [], "loss": [], "accuracy": []}
    for epoch in range(50):
        history = model.fit(X_train, y_train, epochs=1, verbose=1)
        training_history["epochs"].append(epoch + 1)
        training_history["loss"].append(history.history["loss"][0])
        training_history["accuracy"].append(1 - history.history["mae"][0])
        with open(history_path, "w") as f:
            json.dump(training_history, f)

    model.save(model_path)
    print(f"Модель сохранена в {model_path}")

# Метрики модели
@app.route('/metrics', methods=['GET'])
def get_metrics():
    # Рассчитываем метрики
    results = model.evaluate(X_test, y_test, verbose=0)
    loss = results[0]
    mae = results[1]

    # Рассчитываем MAPE
    y_pred = model.predict(X_test, verbose=0)
    y_test_inverse = scaler_y.inverse_transform(y_test)
    y_pred_inverse = scaler_y.inverse_transform(y_pred)
    mape = np.mean(np.abs((y_test_inverse - y_pred_inverse) / y_test_inverse))  # MAPE

    # Пересчитываем точность
    accuracy = 1 - mape

    return jsonify({
        "MSE": round(loss, 6),
        "MAE": round(mae, 6),
        "MAPE": round(mape, 6),
        "Accuracy": round(accuracy, 6)
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

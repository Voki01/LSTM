import numpy as np
import pandas as pd

# Генерация данных
np.random.seed(42)

# Параметры
num_samples = 1000
temperatures = np.random.randint(-40, 41, num_samples)  # Температура от -40 до 40
prices = np.random.randint(1900, 2801, num_samples)  # Цена от 1900 до 2800
competitor_prices = prices + np.random.randint(-100, 201, num_samples)  # Цена конкурентов от 1800 до 3000
discounts = np.random.randint(1, 31, num_samples)  # Скидка от 1% до 30%

# Расчёт спроса
aluminum_radiators = (
    600 - (prices - 1900) * 0.3 + discounts * 6 + (30 - np.abs(temperatures)) * 4
).astype(int)

copper_radiators = (
    350 - (prices - 1900) * 0.2 + discounts * 4 + (30 - np.abs(temperatures)) * 3
).astype(int)

# Убираем отрицательные значения
aluminum_radiators = np.clip(aluminum_radiators, 50, 600)
copper_radiators = np.clip(copper_radiators, 50, 350)

# Формирование таблицы
data = pd.DataFrame({
    "Temperature": temperatures,
    "Price": prices,
    "CompetitorPrice": competitor_prices,
    "Discount": discounts,
    "AluminumRadiators": aluminum_radiators,
    "CopperRadiators": copper_radiators
})

# Сохранение данных в CSV
data.to_csv("radiator_data.csv", index=False)
print("Данные успешно сгенерированы и сохранены в 'radiator_data.csv'.")

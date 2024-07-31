import enum

import matplotlib.pyplot as plt
import numpy as np
import time

from keras import Input
from keras.layers import LSTM, GRU, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

class WeatherParameter(enum.Enum):
    temp = 1
    windSpeed = 2
    pressure = 3


# Чтение из бинарного файла
train_data = np.load('train_data.npy', allow_pickle=True)
test_data = np.load('test_data.npy', allow_pickle=True)

# Преобразование данных в форматы X, y для обучения
def create_dataset(data):
    X, y = [], []
    for i in range(len(data[0])):
        X.append(data[WeatherParameter.temp.value][i])  #Изменить WeatherParameter при необходимости

    window_size = 100
    # Количество подмассивов, которые можно создать
    num_subarrays = len(X) - window_size + 1
    # Создание массива массивов
    res_input = np.array([X[i:i + window_size] for i in range(num_subarrays-1)])
    result = res_input.reshape(len(res_input), 100)

    res_output = np.array([X[i+window_size] for i in range(num_subarrays-1)])
    return np.array(result).astype('float32'), np.array(res_output).astype('float32')

X_train, y_train = create_dataset(train_data)
X_test, y_test = create_dataset(test_data)

# Создание модели
model = Sequential()

model.add(Input(shape=(100, 1)))   # Входной слой
model.add(GRU(57))
model.add(Dense(1))  # Один выходной нейрон

# Компиляция модели
model.compile(optimizer='adam', loss='mean_squared_error')

# Вывод структуры модели
model.summary()

start_time = time.time()
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=5, verbose=1)
end_time = time.time()

elapsed_time = end_time - start_time
# Оценка модели
test_loss = model.evaluate(X_test, y_test, verbose=1)
print(f'Test Loss: {test_loss}')

mae_train = np.mean(np.abs(history.history['loss']))
mae_test = np.mean(np.abs(history.history['val_loss']))
min_train = np.min(np.abs(history.history['loss']))
min_test = np.min(np.abs(history.history['val_loss']))

# Построение графика обучения
plt.plot(history.history['loss'], label='Тренировочная ошибка')
plt.plot(history.history['val_loss'], "--", label='Тестовая ошибка')
plt.xlabel('Эпоха')
plt.ylabel('Величина ошибки')
plt.legend()
plt.show()

print(f'Время работы: {elapsed_time}')
print(f'Средняя ошибка обучения: {mae_train}')
print(f'Средняя ошибка в тестовом наборе: {mae_test}')
print(f'Минимальная ошибка обучения: {min_train}')
print(f'Минимальная ошибка в тестовом наборе: {min_test}')
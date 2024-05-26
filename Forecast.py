import pandas as pd
from datetime import datetime
import numpy as np

# Чтение CSV файла в DataFrame
file_path = 'weatherHistory.csv'
df = pd.read_csv(file_path, usecols=['Formatted Date', 'Temperature (C)', 'Wind Speed (km/h)', 'Pressure (millibars)'])
df = df.transpose()
data_array = df.values


# Вывод первых 5 строк таблицы для проверки
print(df.head())

# Преобразование строк в объекты datetime
date_objects = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f %z') for date in data_array[0]]

data_array[0] = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f %z') for date in data_array[0]]
# Сортировка объектов datetime
date_objects_sorted = sorted(date_objects)

data_array[1] = data_array[1][np.argsort(data_array[0])]
data_array[2] = data_array[2][np.argsort(data_array[0])]
data_array[3] = data_array[3][np.argsort(data_array[0])]

data_array[0] = data_array[0][np.argsort(data_array[0])]
# Преобразование обратно в строки
data_array[0] = [date.strftime('%Y-%m-%d %H:%M:%S.%f %z') for date in date_objects_sorted]

# Сохранение в бинарный файл
np.save('matrix.npy', data_array)

# Чтение из бинарного файла
#loaded_matrix = np.load('matrix.npy', allow_pickle=True)
#print(loaded_matrix)

# Вывод отсортированных дат
print(data_array[0])
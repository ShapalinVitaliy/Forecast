import pandas as pd
from datetime import datetime
import numpy as np
import csv

# Чтение CSV файла в DataFrame
file_path = 'weatherHistory.csv'
df = pd.read_csv(file_path, usecols=['Formatted Date', 'Temperature (C)'])
df = df.transpose()
data_array = df.values


# Вывод первых 5 строк таблицы для проверки
print(df.head())


'''date_strings = [
    "2006-04-01 00:00:00.000 +0200",
    "2006-04-10 00:00:00.000 +0200",
    "2006-04-02 00:00:00.000 +0200",
    "2006-04-03 00:00:00.000 +0200"
]'''


# Преобразование строк в объекты datetime
date_objects = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f %z') for date in data_array[0]]

# Сортировка объектов datetime
date_objects_sorted = sorted(date_objects)

# Преобразование обратно в строки
data_array[0] = [date.strftime('%Y-%m-%d %H:%M:%S.%f %z') for date in date_objects_sorted]

# Вывод отсортированных дат
print(data_array[0])
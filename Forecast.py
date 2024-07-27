import pandas as pd
from datetime import datetime
import numpy as np

# Чтение из бинарного файла
loaded_matrix = np.load('matrix.npy', allow_pickle=True)

train_data = loaded_matrix[:4, 0:26304]
test_data = loaded_matrix[:4, 26304:35040]

# Сохранение в бинарный файл
np.save('train_data.npy', train_data)
np.save('test_data.npy', test_data)
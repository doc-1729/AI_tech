"""
Нормализация данных

"""
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Чтение датасета из файла
df = pd.read_csv('PCA-13-2D.csv')
# Индикация исходного датасета
sns.scatterplot(data = df, x = "PCA1", y = "PCA2", s = 100)
plt.show()


"""
Z-нормировка: центрирование и стандартизация СКО

"""
# Нормировка по столбцам
d = preprocessing.normalize(df, axis=0)

# Преобразование в Dataframe
scaled_df = pd.DataFrame(d, columns=["x1","x2"])

# Индикация
sns.scatterplot(data = scaled_df, x = "x1", y = "x2", s = 100)
plt.show()


"""
Минимаксная нормализация к диапазону [0;1]

"""
# Минимаксная нормализация
scaler = preprocessing.MinMaxScaler()  # Создание скейлера
d = scaler.fit_transform(df)  # Минимаксная нормализация

# Преобразование в Dataframe
scaled_df = pd.DataFrame(d, columns=["x1","x2"])

# Индикация
sns.scatterplot(data = scaled_df, x = "x1", y = "x2", s = 100)
plt.show()


"""
Минимаксная нормализация к произвольному диапазону [xmin;xmax]

"""
# Минимаксная нормализация
scaler = preprocessing.MinMaxScaler(feature_range=(0, 100))  # Инициализация скейлера
d = scaler.fit_transform(df)  # Минимаксная нормализация

# Преобразование в Dataframe
scaled_df = pd.DataFrame(d, columns=["x1","x2"])

# Индикация
sns.scatterplot(data = scaled_df, x = "x1", y = "x2", s = 100)
plt.show()
"""
Метод главных компонент (Principal Component Analysis, PCA)

"""
import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

rnd.seed(10005)  # Инициализация генератора псевдослучайных чисел

# Параметры для генерации выборки y = w0 + w1*x1 + N(0,sigma)
W=[2,3]  # Коэффициенты линейной регрессии
x_min, x_max = 1, 5 # Границы кластера по горизонтали
e_sigma = 2  # СКО нормального шума

#  Функция генерации выборки
def X_1D_generate(rows):  # формирует массив случайных векторов признаков [x1,...,XD] с равномерным законом
    data = []
    for i in range(rows):
        x = rnd.uniform(x_min, x_max)
        data.append([x, W[0] + x*W[1] + rnd.gauss(0,e_sigma)])
    return pd.DataFrame(data, columns=['x1','y'])


# Генерация выборки в DataFrame
df = X_1D_generate(10)
# print(df)

# Метод главных компонент (PCA)
pca = PCA(n_components = 2)  # задание параметров PCA
F = pca.fit_transform(df)  # Новые координаты

# Преобразование массива новых координат в формат Data Frame
df2 = pd.DataFrame (F, columns = ['PCA1','PCA2'])


# Индикация на основе seaborn
mark = {'1': "s", '2': "D", '3': "o"}
sns.scatterplot(data = df, x = "x1", y = "y", s = 100,
                markers = mark, edgecolors = 'black', linewidths = 2)
plt.plot([x_min, x_max], [W[0] + x_min*W[1], W[0] + x_max*W[1]],
         linewidth = 1, color='green', linestyle = '-')
plt.show()

sns.scatterplot(data = df2, x = "PCA1", y = "PCA2", s = 100, vmin = -5, vmax = 5,
                markers = mark, edgecolors = 'black', linewidths = 2)
plt.plot([-6, 8], [0, 0],
         linewidth = 1, color='green', linestyle = '-')
plt.show()


# Индикация на основе matplotlib
fig, ax = plt.subplots()
plt.scatter(df['x1'], df['y'], marker = 'D')
plt.plot([x_min, x_max], [W[0] + x_min*W[1], W[0] + x_max*W[1]],
         linewidth = 1, color='green', linestyle = '-')
ax.set_xlabel('x1', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.grid()  # формирует сетку
ax.set_xlim(0, 6)
ax.set_ylim(2, 18)
plt.show()

fig, ax = plt.subplots()
plt.scatter(df2['PCA1'], df2['PCA2'], marker = 'D')
plt.plot([-6, 6], [0, 0],
         linewidth = 1, color='green', linestyle = '-')
ax.set_xlabel('PCA1', fontsize=12)
ax.set_ylabel('PCA2', fontsize=12)
ax.grid()  # формирует сетку
ax.set_xlim(-8, 8)
ax.set_ylim(-5, 5)
plt.show()
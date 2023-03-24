"""
Генерация и визуализация датасета (2 класса, 2-D пространство признаков)
"""

import random as rnd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

rnd.seed(1000)  # Инициализация генератора псевдослучайных чисел

N = 13  # Число объектов в каждом классе

# Генерация случайных значений признаков объектов класса y=2 для обучающей выборки
m_x1_2 = 6; sigma_x1_2 = 1.5; m_x2_2 = 0; sigma_x2_2 = 0.7
x1_2 = np.asarray([rnd.gauss(m_x1_2, sigma_x1_2) for i in range(N)])
x2_2 = np.asarray([rnd.gauss(m_x2_2, sigma_x2_2) for i in range(N)])
m1_2, m2_2 = np.mean(x1_2), np.mean(x2_2)  # центр тяжести

# Индикация прецедентов второго класса средствами Matplotlib
fig, ax = plt.subplots()

plt.scatter(x1_2, x2_2, marker='D', color='green')  # y=2
plt.scatter(m1_2, m2_2, marker='D', color='limegreen', s=100)  # центр тяжести

# Генерация случайных значений признаков объектов класса y=1 для обучающей выборки
m_x1_1 = 2; sigma_x1_1 = 0.6; m_x2_1 = 5; sigma_x2_1 = 1
x1_1 = np.asarray([rnd.gauss(m_x1_1, sigma_x1_1) for i in range(N)])
x2_1 = np.asarray([rnd.gauss(m_x2_1, sigma_x2_1) for i in range(N)])
m1_1, m2_1 = np.mean(x1_1), np.mean(x2_1)  # центр тяжести

# Индикация прецедентов первого класса средствами Matplotlib
plt.scatter(x1_1, x2_1, marker='s', color='dodgerblue')  # y=1
plt.scatter(m1_1, m2_1, marker='D', color='blue', s=100)  # центр тяжести

# задание признаков наблюдаемого объекта
x1_nabl = 5; x2_nabl = 3
plt.scatter([x1_nabl], [x2_nabl], marker='o', color='red', s=100)  # наблюдаемый образ

# Оформление графика
plt.text(1.5, 6.5, s=r'$X_1^{13}=\{\mathbf{x}_1^{(1)}, \mathbf{x}_2^{(1)},..., \mathbf{x}_{13}^{(1)} \}$', fontsize=12,
         bbox=dict(color='w'), rotation=0)
plt.text(0.5, 5, s=r'$y=1$', fontsize=12, bbox=dict(color='w'), rotation=0)
plt.text(1.5, -1.5, s=r'$X_2^{13}=\{\mathbf{x}_1^{(2)}, \mathbf{x}_2^{(2)},..., \mathbf{x}_{13}^{(2)} \}$', fontsize=12,
         bbox=dict(color='w'), rotation=0)
plt.text(3.2, 1, s=r'$y=2$', fontsize=12, bbox=dict(color='w'), rotation=0)
plt.text(5.3, 4, s=r'$\mathbf{x}$', fontsize=12, bbox=dict(color='w'), rotation=0)

ax.set_xlabel(r'$признак _ x_1$', fontsize=12)
ax.set_ylabel(r'$признак _ x_2$', fontsize=12)
ax.grid()  # формирует сетку
ax.set_xlim(0, 8)
ax.set_ylim(-2, 8)

plt.show()

"""
Преобразование обучающей выборки к формату DataFrame
"""

data = np.ones((N, 4))
df1 = pd.DataFrame(data, columns=['x1', 'x2', 'class', 'distanse'], index=range(N))
df1['x1'] = x1_1
df1['x2'] = x2_1
df1['class'] = 'y=1'
df1['distanse'] = float('inf')
df2 = pd.DataFrame(data, columns=['x1', 'x2', 'class', 'distanse'], index=range(N, 2 * N))
df2['x1'] = x1_2
df2['x2'] = x2_2
df2['class'] = 'y=2'
df2['distanse'] = float('inf')
df = pd.concat([df1, df2], axis=0)
print(df)

# запись DataFrame в файл *.csv
# df.to_csv('x1_x2_class_distanse_2x13.csv',index=False)


"""
Визуализация набора данных из файла *.csv 
"""
import pandas as pd
import seaborn as sns

# Чтение набора данных из файла в DataFrame
df = pd.read_csv('x1_x2_class_distanse_2x13.csv')

# Визуализаци DataFrame средствами продвинутой графики seaborn
mark = {'y=1': "s", "y=2": "D"}
sns.scatterplot(data=df, x="x1", y="x2", hue="class", style="class", s=100, markers=mark)
plt.show()

"""
Визуализация готового набора данных о цветке Ирис 
из учебного репозитория
https://archive.ics.uci.edu/ml/datasets/Iris
з класса, 4 признака
"""

# Чтение набора данных из файла
iris_data = pd.read_csv('iris.csv')
# iris_data.head()

# Разделение набора данных на три DataFrame классов
df_y1 = iris_data[iris_data["class"] == "Iris-setosa"]
df_y2 = iris_data[iris_data["class"] == "Iris-versicolor"]
df_y3 = iris_data[iris_data["class"] == "Iris-virginica"]

# Визуализация четырехмерного признакового пространства в двумерных проекциях
fig = plt.figure(figsize=(16, 9))

ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)
ax6 = fig.add_subplot(2, 3, 6)

a = "x1"; b = "x2"
ax1.scatter(df_y1[a], df_y1[b])
ax1.scatter(df_y2[a], df_y2[b])
ax1.scatter(df_y3[a], df_y3[b])
ax1.set_xlabel(a)
ax1.set_ylabel(b)

a = "x1"; b = "x3"
ax2.scatter(df_y1[a], df_y1[b])
ax2.scatter(df_y2[a], df_y2[b])
ax2.scatter(df_y3[a], df_y3[b])
ax2.set_xlabel(a)
ax2.set_ylabel(b)

a = "x1"; b = "x4"
ax3.scatter(df_y1[a], df_y1[b])
ax3.scatter(df_y2[a], df_y2[b])
ax3.scatter(df_y3[a], df_y3[b])
ax3.set_xlabel(a)
ax3.set_ylabel(b)

a = "x2"; b = "x3"
ax4.scatter(df_y1[a], df_y1[b])
ax4.scatter(df_y2[a], df_y2[b])
ax4.scatter(df_y3[a], df_y3[b])
ax4.set_xlabel(a)
ax4.set_ylabel(b)

a = "x2"; b = "x4"
ax5.scatter(df_y1[a], df_y1[b])
ax5.scatter(df_y2[a], df_y2[b])
ax5.scatter(df_y3[a], df_y3[b])
ax5.set_xlabel(a)
ax5.set_ylabel(b)

a = "x3"; b = "x4"
ax6.scatter(df_y1[a], df_y1[b])
ax6.scatter(df_y2[a], df_y2[b])
ax6.scatter(df_y3[a], df_y3[b])
ax6.set_xlabel(a)
ax6.set_ylabel(b)

plt.show()

"""
Проекции 4D признакового пространства в 3D
"""
# from mpl_toolkits import mplot3d

# Чтение набора данных из файла
iris_data = pd.read_csv('iris.csv')

# Разделение набора данных на три DataFrame классов
df_y1 = iris_data[iris_data["class"] == "Iris-setosa"]
df_y2 = iris_data[iris_data["class"] == "Iris-versicolor"]
df_y3 = iris_data[iris_data["class"] == "Iris-virginica"]

# Визуализация четырехмерного признакового пространства в 3D проекциях
fig = plt.figure(figsize=(16, 9))
ax = plt.axes(projection='3d')

a = "x1"; b = "x3"; c = "x2"
ax.scatter3D(df_y1[a], df_y1[b], df_y1[c], s=100)
ax.scatter3D(df_y2[a], df_y2[b], df_y1[c], s=100)
ax.scatter3D(df_y3[a], df_y3[b], df_y1[c], s=100)
ax.set_xlabel(a, fontweight='bold')
ax.set_ylabel(b, fontweight='bold')
ax.set_zlabel(c, fontweight='bold')

fig = plt.figure(figsize=(16, 9))
ax1 = plt.axes(projection='3d')

a = "x4"; b = "x3"; c = "x1"
ax1.scatter3D(df_y1[a], df_y1[b], df_y1[c], s=100)
ax1.scatter3D(df_y2[a], df_y2[b], df_y1[c], s=100)
ax1.scatter3D(df_y3[a], df_y3[b], df_y1[c], s=100)
ax1.set_xlabel(a, fontweight='bold')
ax1.set_ylabel(b, fontweight='bold')
ax1.set_zlabel(c, fontweight='bold')

fig = plt.figure(figsize=(16, 9))
ax2 = plt.axes(projection='3d')

a = "x4"; b = "x1"; c = "x2"
ax2.scatter3D(df_y1[a], df_y1[b], df_y1[c], s=100)
ax2.scatter3D(df_y2[a], df_y2[b], df_y1[c], s=100)
ax2.scatter3D(df_y3[a], df_y3[b], df_y1[c], s=100)
ax2.set_xlabel(a, fontweight='bold')
ax2.set_ylabel(b, fontweight='bold')
ax2.set_zlabel(c, fontweight='bold')

fig = plt.figure(figsize=(16, 9))
ax3 = plt.axes(projection='3d')

a = "x4"; b = "x2"; c = "x3"
ax3.scatter3D(df_y1[a], df_y1[b], df_y1[c], s=100)
ax3.scatter3D(df_y2[a], df_y2[b], df_y1[c], s=100)
ax3.scatter3D(df_y3[a], df_y3[b], df_y1[c], s=100)
ax3.set_xlabel(a, fontweight='bold')
ax3.set_ylabel(b, fontweight='bold')
ax3.set_zlabel(c, fontweight='bold')

plt.show()

"""
Визуализация набора данных "объекты-признаки" в параллельных координатах
"""
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import parallel_coordinates

data = pd.read_csv('iris.csv')
data_1 = data[['class', 'x1', 'x2', 'x3', 'x4']]

fig = plt.figure()
parallel_coordinates(data_1, 'class')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fancybox=True, shadow=True)
plt.show()

"""
Отображение вектор-признаков многомерного пространства
с помощью лепестковой диаграммы
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Чтение набора данных из файла
df = pd.read_csv('iris.csv')
print(df)

# Извлечение по одному объекту каждого класса из DataFrame
labels = np.asarray(["x1", "x2", "x3", "x4"])
stats = df.loc[0, labels].values
stats1 = df.loc[80, labels].values
stats2 = df.loc[120, labels].values

# Подготовка параметров для лепестковой диаграммы
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
stats = np.concatenate((stats, [stats[0]]))
stats1 = np.concatenate((stats1, [stats1[0]]))
stats2 = np.concatenate((stats2, [stats2[0]]))
angles = np.concatenate((angles, [angles[0]]))
labels = np.concatenate((labels, [labels[0]]))

# Построение лепестковой диаграммы
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)

ax.plot(angles, stats, 'o-', linewidth=2, label='Iris Setosa')
ax.plot(angles, stats1, 'o-', linewidth=2, label='Iris Versicolour')
ax.plot(angles, stats2, 'o-', linewidth=2, label='Iris Virginica')
# ax.fill(angles, stats, alpha=0.25)
ax.set_thetagrids(angles * 180 / np.pi, labels)
# ax.set_title([df.loc[0,"class"]])
ax.grid(True)
ax.legend()

plt.show()
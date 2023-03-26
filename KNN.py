"""
KNN - метод "k ближайших соседей"
(состав файла:
1. Программа генерации обучающей выборки для двух классов по N прецедентов в 2D пространстве признаков;
2. Программа распознавания наблюдаемого образа на основе алгоритма KNN (программированик с нуля);
3. Программа генерации обучающей выборки для 3-х классов по N прецедентов в 2D пространстве признаков;
4. Программа распознавания наблюдаемого образа на основе алгоритма KNN с формированием карты областей классов
   (программированик с нуля).
5. Программа распознавания методом KNN  средствами библиотеки "sklearn"
"""

import math
import random as rnd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
1. Программа генерации обучающей выборки для двух классов по N прецедентов в 2D пространстве признаков;
"""
rnd.seed(1000)  # Инициализация генератора сучайных чисел
N = 13  # Число объектов каждого класса
m_x1_2 = 6; sigma_x1_2 = 1.5; m_x2_2 = 0; sigma_x2_2 = 0.7  # Параметры генерации случайных кауссовских координат

# Генерация объектов второго класса
x1_2 = np.asarray([rnd.gauss(m_x1_2, sigma_x1_2) for i in range(N)])
x2_2 = np.asarray([rnd.gauss(m_x2_2, sigma_x2_2) for i in range(N)])
m1_2, m2_2 = np.mean(x1_2), np.mean(x2_2)  # центр тяжести кластера

# Генерация объектов первого класса
m_x1_1 = 2; sigma_x1_1 = 0.6; m_x2_1 = 5; sigma_x2_1 = 1
x1_1 = np.asarray([rnd.gauss(m_x1_1, sigma_x1_1) for i in range(N)])
x2_1 = np.asarray([rnd.gauss(m_x2_1, sigma_x2_1) for i in range(N)])
m1_1, m2_1 = np.mean(x1_1), np.mean(x2_1) # центр тяжести

# Индикация обучающей выборки
fig, ax = plt.subplots()
plt.scatter(x1_2, x2_2, marker = 'D', color = 'green')  # y=2
plt.scatter(m1_2, m2_2, marker = 'D', color = 'limegreen', s = 100)  # центр тяжести
plt.scatter(x1_1, x2_1, marker = 's', color = 'dodgerblue')  # y=1
plt.scatter(m1_1, m2_1, marker = 'D', color = 'blue', s = 100)  # центр тяжести

plt.text(1.5, 6.5, s = r'$X_1^{13}=\{\mathbf{x}_1^{(1)}, \mathbf{x}_2^{(1)},..., \mathbf{x}_{13}^{(1)} \}$', fontsize=12, bbox=dict(color='w'), rotation=0)
plt.text(0.5, 5, s = r'$y=1$', fontsize=12, bbox=dict(color='w'), rotation=0)
plt.text(1.5, -1.5, s = r'$X_2^{13}=\{\mathbf{x}_1^{(2)}, \mathbf{x}_2^{(2)},..., \mathbf{x}_{13}^{(2)} \}$', fontsize=12, bbox=dict(color='w'), rotation=0)
plt.text(3.2, 1, s = r'$y=2$', fontsize=12, bbox=dict(color='w'), rotation=0)
plt.text(5.3, 4, s = r'$\mathbf{x}$', fontsize=12, bbox=dict(color='w'), rotation=0)

ax.set_xlabel(r'$признак _ x_1$', fontsize=12)
ax.set_ylabel(r'$признак _ x_2$', fontsize=12)
ax.grid()  # формирует сетку
ax.set_xlim(-2, 8)
ax.set_ylim(-2, 8)



"""
2. Программа распознавания наблюдаемого образа на основе алгоритма KNN (программированик с нуля).

"""
# Функция вычисления расстояния меду двумя точками в двумерном пространстве
def distance(p1, p2):
    return math.sqrt( ((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2) )

# Функция списка расстояний от наблюдаемого образа х до каждой точки одного из кластеров
def list_of_distanse(x, x1, x2):
    return np.asarray([distance(x, [x1[i], x2[i]]) for i in range(len(x1))])

# Преобразование массивов координат кластеров в датасеты формата DataFrame
data = np.ones((N, 4))
df1 = pd.DataFrame(data, columns=['x1','x2','class','distanse'],index=range(N))
df1['x1'] = x1_1; df1['x2'] = x2_1; df1['class'] = 'y=1'; df1['distanse'] = float('inf')

df2 = pd.DataFrame(data, columns=['x1','x2','class','distanse'],index=range(N,2*N))
df2['x1'] = x1_2; df2['x2'] = x2_2; df2['class'] = 'y=2'; df2['distanse'] = float('inf')

df = pd.concat([df1, df2], axis=0)  # Общий датасет двух классов
#print(df)
#df.to_csv('x1_x2_class_distanse_2x13.csv',index=False)  # запись в файл

x1_nabl, x2_nabl = 5, 3  # Задание признаков распознаваемого объекта

# Вычисляем массив расстоний от наблюдаемого образа х до каждой точки обучающей выборки
r = list_of_distanse([x1_nabl, x2_nabl], df['x1'].values, df['x2'].values)
ri = [[i, r[i]] for i in range(len(r))]  # добавляем колонку индексов
ri.sort(key=lambda x: x[1])  # сортируем расстояния по возрастанию
# print(ri)
# minIndex = np.argmin(r)

# Индикация k ближайших соседей
k_NN = 5  #  число ближайших соседей
for i in range(k_NN):
    x1 = [df['x1'][ri[i][0]], x1_nabl]
    x2 = [df['x2'][ri[i][0]], x2_nabl]
    plt.plot(x1, x2, marker = '.', color = 'y', linewidth = '2', linestyle = "-")

# Прорисовка наблюдаемого образа
plt.scatter([x1_nabl], [x2_nabl], marker = 'o', color = 'red', s = 100)  # наблюдаемый образ
plt.text(-1, 1.8, s = r'kNN: $k=5$', fontsize=12, bbox=dict(color='w'), rotation=0)

plt.show()

"""
3. Программа генерации обучающей выборки 
для 3-х классов по N прецедентов в 2D пространстве признаков

"""
import numpy as np
import matplotlib.pyplot as plt
# import scipy.stats as sps
import random as rnd
import pandas as pd
import seaborn as sns

rnd.seed(1000)

# Функция генерации обучающей выборки для трех классов
def norm_2D_classes_generate(rows):
    m_s_xy = [[2, 1, 2, 1], [8, 2, 3, 1], [6, 1, 7, 1]]  # в каждой строке M_x,Sig_x, M_y, Sig_y
    n_classes = len(m_s_xy)
    data = []
    for classNum in range(n_classes):
        for i in range(rows):
            data.append([rnd.gauss(m_s_xy[classNum][0], m_s_xy[classNum][1]),
                         rnd.gauss(m_s_xy[classNum][2], m_s_xy[classNum][3]), 'y=' + str(classNum + 1), classNum])
    return pd.DataFrame(data, columns=['x1', 'x2', 'class', 'classNum'])


# Генерации обучающей выборки для трех классов
N = 30  # Число прецедентов в каждом классе
df = norm_2D_classes_generate(N)
# df.to_csv('x1_x2_dataset_3x30.csv',index=False)  # запись в файл

# Индикация обучающей выборки в 2D пространстве признаков
mark = {'y=1': "s", 'y=2': "D", 'y=3': "o"}
sns.scatterplot(data=df, x="x1", y="x2",
                hue="class", style="class", s=100,
                markers=mark, edgecolors='black', linewidths=2)
plt.show()

"""
4. Программа распознавания наблюдаемого образа на основе алгоритма KNN с формированием карты областей классов
   (программированик с нуля).
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


# Генератор координатной сетки
def generateTestMesh(df):
    x_min = min(df.x1) - 1.0
    x_max = max(df.x1) + 1.0
    y_min = min(df.x2) - 1.0
    y_max = max(df.x2) + 1.0
    h = 0.1  # Шаг координатной сетки
    XX, YY = np.arange(x_min, x_max, h), np.arange(y_min, y_max, h)
    return np.meshgrid(XX, YY)


# Классификатор KNN. Возвращает массив номеров классов
def ClassifikatorKNN(df, testData, k):
    def dist(a, bx, by):
        return np.sqrt((a[0] - bx) ** 2 + (a[1] - by) ** 2)

    nClasses = len(pd.unique(df['class']))
    testLabels = []
    for testPoint in testData:
        # Определение k ближайших обучающих образов из df к одному из тестовых данных
        stat = [0 for i in range(nClasses)]
        R = sorted([[dist(testPoint, df.x1[i], df.x2[i]), df['classNum'][i]] for i in range(len(df))])[0:k]
        for r in R:
            stat[r[1]] += 1
        testLabels.append(np.argmax(stat))
    return testLabels


# Функция индикации
def show_KNN(testMesh, testLabels, k):
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # Задание цветовой палитры карты  трех классов
    cmap_name = 'my_list'  # Задание имени цветовой карты
    testColormap = LinearSegmentedColormap.from_list(cmap_name, colors)
    plt.pcolormesh(testMesh[0], testMesh[1],
                   np.asarray(testLabels).reshape(testMesh[0].shape), cmap=testColormap, alpha=0.2)
    mark = {'y=1': "s", 'y=2': "D", 'y=3': "o"}  # Задание маркеров для трех кластеров
    sns.scatterplot(data=df, x="x1", y="x2", hue="class", style="class", s=100,
                    markers=mark, edgecolors='black', linewidths=2)
    plt.text(0, 8.5, s=r'$k=$' + str(k), fontsize=12, bbox=dict(color='w'))


# KNN с формированием крты областей классов
df = pd.read_csv('x1_x2_dataset_3x30.csv')  # Чтение набора данных из файла в DataFrame
nClasses = len(pd.unique(df['class']))  # Определение числа классов в обучающем DataFrame
k_NN = 3  # Параметр алгоритма KNN - число ближайших соседей
testMesh = generateTestMesh(df)  # Создание координатной сетки тестовых данных
testMeshPoints = zip(testMesh[0].ravel(), testMesh[1].ravel())  # Преобразование сетки в список пар координат
testLabels = ClassifikatorKNN(df, testMeshPoints, k_NN)  # Формирование массива с результатами распознавания
show_KNN(testMesh, testLabels, k_NN)  # Индикация результатов

plt.show()



"""
5. Программа распознавания методом KNN  средствами библиотеки "sklearn"
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Исходные данные]
df = pd.read_csv('x1_x2_dataset_3x30.csv')  # Чтение обучающей выборки из файла в DataFrame
K_NN = 3  # число ближайших соседей
x_nabl = [(8, 4.5)]  # Задание координат распознаваемого объекта

# Обучение классификатора KNN
data = df[['x1','x2']].to_numpy(copy=True)
classes = list(df['class'])
knn = KNeighborsClassifier(n_neighbors=K_NN)  # Инициализация классификатора KNN
knn.fit(data, classes)  # Обучение классификатора KNN

# Распознавание
prediction = knn.predict(x_nabl)  # Возвращает номер класса наблюдаемого объекта

# Индикация обучающей выборки
mark = {'y=1': "s", 'y=2': "D", 'y=3': "o"}  # Задание маркеров для трех кластеров
sns.scatterplot(data = df, x = "x1", y = "x2", hue = "class", style = "class", s = 100,
                    markers = mark, edgecolors = 'black', linewidths = 2)
plt.text(0, 8.5, s = r'$k=$'+str(K_NN), fontsize=12, bbox=dict(color='w'))

# Индикация результатов распознавания
plt.scatter([x_nabl[0][0]], [x_nabl[0][1]], marker = 'P', color = 'red', edgecolors = 'black', s = 100)  # наблюдаемый образ)
plt.text(x=x_nabl[0][0]-1, y=x_nabl[0][1]+0.7, s=f"new x: {prediction[0]}", fontsize=12, bbox=dict(color='w'))

plt.show()

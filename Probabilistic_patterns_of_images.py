# Вероятностные модели образов

"""
Условные распределения / плотности вероятностей двух признаков объектов

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Чтение обучающей выборки из файла в DataFrame
df = pd.read_csv('x1_x2_class_distanse_2x13.csv')

# Визуализация выборки в двумерном пространстве признаков
#          и построение гистограмм частот условных распределений признаков классов
fig = plt.figure()
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
ax_main = fig.add_subplot(grid[:-1, :-1])
mark = {"y=1": "s", "y=2": "D"}
#print(df)
sns.scatterplot(data = df, x = "x1", y = "x2", hue = "class", s = 100, markers = mark)

ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
sns.histplot(data = df, y = 'x2', bins=10, hue = "class", kde=True, legend=False)

ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])
sns.histplot(data = df, x = 'x1', bins=10, hue = "class", kde=True, legend=False)  # уровиень axes

plt.show()



"""
Безусловные законы распределения признаков двух признаков объектов

"""
# Чтение обучающей выборки из файла в DataFrame
df = pd.read_csv('x1_x2_class_distanse_2x13.csv')

# Визуализация выборки в двумерном пространстве признаков
#          и построение гистограмм частот условных распределений признаков классов
fig = plt.figure()
grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.2)
ax_main = fig.add_subplot(grid[:-1, :-1])
mark = {"y=1": "s", "y=2": "D"}
sns.scatterplot(data = df, x = "x1", y = "x2", s = 100, markers = mark)

ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
sns.histplot(data = df, y = 'x2', bins=10, kde=True, legend=False)

ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])
sns.histplot(data = df, x = 'x1', bins=10, kde=True, legend=False)  # уровиень axes

plt.show()


"""
Более короткие спсобы визуализации условных и безусловных распределений признаков
средствами Seaborn

"""
# Визуализация условных выборочных распределений признаков
fig = plt.figure()
sns.jointplot(data=df, x="x1", y="x2", hue="class")
plt.show()

# Визуализация условных выборочных распределений признаков
fig = plt.figure()
sns.jointplot(data=df, x="x1", y="x2",)
plt.show()


"""
Плотности вероятности признаков для трех классов четырехмерного 
признакового пространства

"""
import pandas as pd
import seaborn as sns

# Чтение обучающей выборки из файла в DataFrame
df = pd.read_csv('iris.csv')

# Построение всех проекций четырехмерного признакового пространства на плоскость
#            и условных выборочных законов распределения признаков
plt.figure(figsize=(10,8), dpi= 80)
sns.pairplot(df, kind="reg", hue="class")
plt.show()



"""
Гауссовские условные распределения признака для двух классов
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps

# Функция вычисления плотности нормального распределения
def f_norm(x, m = 0.0, s = 1.0):
    a = math.sqrt(2 * np.pi) * s
    b = math.exp(- (x - m)**2 / (2 * s**2))
    return  b / a

# Формирование массивов координат x и y для построения условных распределений
xmin = -10.0; xmax = 15.0; dx = 0.01
xlist = np.arange(xmin, xmax, dx)
ylist = np.asarray([f_norm(x, 0, 3.0) for x in xlist])
y2list = np.asarray([f_norm(x, 5, 3.0) for x in xlist])

# наблюдаемое значение признака
x_nabl = 1.0

# Подготовка индикации линии наблюдаемого объекта
p1_x = f_norm(x_nabl, 0, 3.0)
p2_x = f_norm(x_nabl, 5, 3.0)
p_x = np.full(3,x_nabl)  # заполнение массива х для построения линии
p_y = np.asarray([0, p2_x, p1_x])  # масиив у для маркеров линии

# Визуализация графиков
fig, ax = plt.subplots()

plt.fill_between(xlist, y2list, where = (ylist <= y2list), alpha = 0.4)
plt.fill_between(xlist, ylist, alpha = 0.4)
plt.plot(xlist, y2list, linewidth = 3)
plt.plot(xlist, ylist, linewidth = 3)  # линейный график
plt.plot(p_x, p_y, linewidth = 2, marker = 'o', linestyle = "--", markersize = 8)

ax.set(xlabel = 'значение признака $x$')
ax.set_ylabel('вероятность / плотность')  # можно так
# fig.suptitle('suptitle', fontsize=16)
plt.text(-7, 0.14, s = r'$p(x|y=1)$', fontsize=12, bbox=dict(color='w'), rotation=0)
plt.text(7, 0.14, s = r'$p(x|y=2)$', fontsize=12, bbox=dict(color='w'), rotation=0)
# plt.text(3.5, 0.02, s = r'$\alpha$', fontsize=12, bbox=dict(color='w'), rotation=0)
# plt.text(1.1, 0.02, s = r'$\beta$', fontsize=12, bbox=dict(color='w'), rotation=0)
ax.grid()  # формирует сетку
# ax.legend(['y1', 'y2', 'y3'], loc='upper center', shadow=True)

ax.set_xlim(xmin, xmax)
ax.set_ylim(0, 0.18)

plt.show()



"""
Многомерное распределение вероятностеq

"""
import math
import matplotlib.pyplot as plt
import numpy as np

# Параметры двумерных условных гауссовских распределений
par1 = [2, 5, 3, 1.5]
par2 = [6, 0, 3, 1.5]

# Формирование сетки для построения 3D поверхностей
x = np.arange(-4, 12, 0.1)
y = np.arange(-4, 9, 0.1)
X, Y = np.meshgrid(x, y)

# Функция двумерного гауссовского распределения
def f2Dnorm(x, y, mx = 0.0, my = 0, sx = 1.0, sy = 1.0):
    a = 2 * np.pi * sx * sy
    b = np.exp(- (((x - mx) / sx)**2 +((y - my) / sy)**2) / 2)
    return  b / a

# Функция безусловного двумерного распределения
def twoNoerm(x,y, par1, par2):
    z = f2Dnorm(x,y, par1[0], par1[1], par1[2], par1[3]) + f2Dnorm(x,y, par2[0], par2[1], par2[2], par2[3])
    return z

# Построение графика двумерного распределения в виде цветового рельефа и линий уровня
fig = plt.figure(figsize = (8, 6))
ax = fig.add_subplot(1,1,1)
C = plt.contour(X,Y,twoNoerm(X,Y, par1, par2),8,colors='black')
plt.contourf(X,Y,twoNoerm(X,Y, par1, par2),8)
plt.clabel(C, inline=1, fontsize=10)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title(r'$p(\mathbf{x})=p(x_1,x_2)$')
plt.show()

# Построение графика двумерного распределения в 3D
Z = twoNoerm(X,Y, par1, par2)
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                 cmap='viridis')
ax.plot_wireframe(X, Y, Z, alpha=0.8)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel('p(x)')

ax.view_init(20, 50)  # изменение ракурса графика

plt.show()
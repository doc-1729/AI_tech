"""
Линейный дискриминантный анализ:
1. Иллюстрация сути LDA
2. Непересекающиеся кластеры
3. Пересекающиеся кластеры
4. Непересекающиеся линейно неразделимые кластеры
5. Иллюстрации к принципу работы линейного байесовского классификатора.
   Одинаковые дисперсии признаков у разных классов.
6. Иллюстрации к принципу работы линейного байесовского классификатора.
   Критерий максимуа апостериарной вероятности.
   Одинаковые дисперсии признаков у разных классов.
7. Иллюстрации к принципу работы линейного байесовского классификатора.
   Критерий максимального правдоподобия (МП) при разных дисперсиях признаков в классе.
   Одинаковые дисперсии у разных классов.
8. Иллюстрации к принципу работы линейного байесовского классификатора.
   Критерий максимального правдоподобия (МП) при одинаковых дисперсиях признаков в классе.
   Разные дисперсии у классов.
9. Принцип разделения линейно неразделимых классов
10. Принцип работы метода потенциальных функция
"""

"""
1. Иллюстрация сути LDA

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('x1_x2_class_distanse_2x13.csv')
mark = {'y=1': "s", "y=2": "D"}
#print(df)
sns.scatterplot(data = df, x = "x1", y = "x2", hue = "class", style = "class", s = 100, markers = mark)
plt.plot([1,6], [-1,5], linewidth = 2, color='black')
plt.text(4, 3, s = r'$f(x)=0$', fontsize=12, bbox=dict(color='w'), rotation=0)
plt.text(1.5, 2, s = r'$f(x)>0$', fontsize=12, bbox=dict(color='w'), rotation=0)
plt.text(2.5, -0.3, s = r'$f(x)<0$', fontsize=12, bbox=dict(color='w'), rotation=0)
plt.show()


"""
2. Непересекающиеся кластеры
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rnd
import seaborn as sns

rnd.seed(10003)

#  Функция генерация кластеров
def norm_2D_classes_generate(rows):
    m_s_xy = [[8,1,3,1],[3,1,6,1]]  # в каждой строке M_x,Sig_x, M_y, Sig_y
    n_classes = len(m_s_xy)
    data = []
    for classNum in range(n_classes):
        for i in range(rows):
            data.append([rnd.gauss(m_s_xy[classNum][0], m_s_xy[classNum][1]), rnd.gauss(m_s_xy[classNum][2], m_s_xy[classNum][3]), str(classNum + 1), classNum])
    return pd.DataFrame(data, columns=['x1','x2','class', 'classNum'])

# Генарция датасета
N = 50  # Число прецедентов каждого класса
df = norm_2D_classes_generate(N)
#df.to_csv('x1_x2_class_distanse_2x50.csv',index=False)  # запись в файл

# Индикация неперсекающихся кластеров
mark = {'1': "s", '2': "D", '3': "o"}
sns.scatterplot(data = df, x = "x1", y = "x2",
                hue = "class", style = "class", s = 100,
                markers = mark, edgecolors = 'black', linewidths = 2)
plt.show()

"""
3. Пересекающиеся кластеры

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rnd
import seaborn as sns

rnd.seed(10003)


#  Функция генерация кластеров
def norm_2D_classes_generate(rows):
    m_s_xy = [[8, 2, 3, 2], [3, 2, 6, 2]]  # в каждой строке M_x,Sig_x, M_y, Sig_y
    n_classes = len(m_s_xy)
    data = []
    for classNum in range(n_classes):
        for i in range(rows):
            data.append([rnd.gauss(m_s_xy[classNum][0], m_s_xy[classNum][1]),
                         rnd.gauss(m_s_xy[classNum][2], m_s_xy[classNum][3]), str(classNum + 1), classNum])
    return pd.DataFrame(data, columns=['x1', 'x2', 'class', 'classNum'])


# Генарция датасета
N = 60  # Число прецедентов каждого класса
df = norm_2D_classes_generate(N)
# df.to_csv('x1_x2_class_distanse_2x60.csv',index=False)  # запись в файл

# Индикация неперсекающихся кластеров
mark = {'1': "s", '2': "D", '3': "o"}
sns.scatterplot(data=df, x="x1", y="x2",
                hue="class", style="class", s=100,
                markers=mark, edgecolors='black', linewidths=2)
plt.show()

"""
4. Непересекающиеся линейно неразделимые кластеры

"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rnd
import pandas as pd
import seaborn as sns

rnd.seed(10003)


#  Функция генерация кластеров
def sin_cos_2D_classes_generate(rows):
    xmin, xmax = 1, 360
    y_sigma1 = 0.3
    y_sigma2 = 0.4
    scale = 1
    sm_y = - 2.3
    data = []
    x = rnd.uniform(xmin, xmax, rows)
    for i in range(rows):
        data.append([x[i], np.sin(scale * x[i] * np.pi / 180) + rnd.normal(0, y_sigma1), 'y=1', 0])
    x = rnd.uniform(xmin, xmax, rows)
    for i in range(rows):
        data.append(
            [x[i], np.cos(scale * x[i] * np.pi / 180 - 90 * np.pi / 180) + rnd.normal(sm_y, y_sigma2), 'y=2', 1])
    return pd.DataFrame(data, columns=['x1', 'x2', 'class', 'classNum'])


# Генарция датасета
N = 60  # Число прецедентов каждого класса
df = sin_cos_2D_classes_generate(60)
a = {"y=1": 'y= +1', "y=2": 'y= - 1'}
df["classes"] = df["class"].map(a)
df["sign"] = - df["classNum"] * 2 + 1
# df.to_csv('sun_cos_clusters_2x60.csv',index=False)  # запись в файл

# Индикация неперсекающихся кластеров
mark = {'y=1': "s", 'y=2': "D"}
sns.scatterplot(data=df, x="x1", y="x2",
                hue="class", style="class", s=100,
                markers=mark, edgecolors='black', linewidths=2)
plt.show()



"""
5. Иллюстрации к принципу работы линейного байесовского классификатора.
Одинаковые дисперсии признаков у разных классов.

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Параметры условных распределений вектора признаков для 2D просртнраства
par1 = [2, 5, 2, 2]  # Математическое ожидание и СКО распределения 1-го класса
par2 = [5, 0, 2, 2]  #  Математическое ожидание и СКО распределения 1-го класса

# Создание координатной сетки для построения двумерных законов распределений классов
x = np.arange(-4, 12, 0.1)
y = np.arange(-4, 9, 0.1)
X, Y = np.meshgrid(x, y)

# Двумерная гауссовская функция
def f2Dnorm(x, y, mx = 0.0, my = 0, sx = 1.0, sy = 1.0):
    a = 2 * np.pi * sx * sy
    b = np.exp(- (((x - mx) / sx)**2 +((y - my) / sy)**2) / 2)
    return  b / a

# Функция разности условных законов распределений двух классов для алгоритма максимального правдоподобия
def twoNorm(x,y, par1, par2):
    z = f2Dnorm(x,y, par1[0], par1[1], par1[2], par1[3]) - f2Dnorm(x,y, par2[0], par2[1], par2[2], par2[3])
#     z = 0.6*f2Dnorm(x,y, par1[0], par1[1], par1[2], par1[3]) - 0.4*f2Dnorm(x,y, par2[0], par2[1], par2[2], par2[3])
    return z

# Индикация принципа формирования дискриминационной функции на плоскости
fig = plt.figure(figsize = (8, 6))
ax = fig.add_subplot(1,1,1)
C = plt.contour(X,Y,twoNorm(X,Y, par1, par2),8,colors='black')
# plt.contourf(X,Y,twoNorm(X,Y, par1, par2),8)
plt.clabel(C, inline=1, fontsize=10)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title(r'$f(\mathbf{x})=p_1(\mathbf{x})-p_2(\mathbf{x})$')
plt.text(8, 7, s = r'$f(\mathbf{x})>0$', fontsize=12, bbox=dict(color='w'))
plt.text(-2, -2, s = r'$f(\mathbf{x})<0$', fontsize=12, bbox=dict(color='w'))
plt.text(par1[0]-0.3, par1[1], s = r'$\mathbf{\mu}_1$', fontsize=12, bbox=dict(color='w'))
plt.text(par2[0]-0.3, par2[1], s = r'$\mathbf{\mu}_2$', fontsize=12, bbox=dict(color='w'))
plt.show()

# fig.savefig('ML_LDA.png', transparent=True, dpi = 100)  # Можно в любом месте; transparent - прозрачность
# fig.savefig('ML_LDA.png', transparent=True, dpi = 100)  # Можно в любом месте; transparent - прозрачность

# Индикация принципа формирования дискриминационной функции в 3D
Z = twoNorm(X,Y, par1, par2)
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                 cmap='viridis')
ax.plot_wireframe(X, Y, Z, alpha=0.8)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel('p(x)')

ax.view_init(20, -150)  # изменение ракурса графика

# plt.savefig('Binorm.png', transparent=True)  #Ддолжно использоваться до plt.show(), иначе пустой файл.
plt.show()
# fig.savefig('Binorm.png', transparent=True, dpi = 100)  # Можно в любом месте; transparent - прозрачность



"""
6. Иллюстрации к принципу работы линейного байесовского классификатора.
Критерий максимуа апостериарной вероятности.
Одинаковые дисперсии признаков у разных классов.

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

# Формирование двумерных условных законов распределений
x, y = np.mgrid[-4:12:.01, -4:9:.1]  # Формирование координатной сетки
pos = np.dstack((x, y))  #Конкатеннация двух одномерных массивов в двумерный
rv1 = multivariate_normal([2, 5], [[2**2, 0], [0, 2**2]])  # Построение закона рапсределения 1-го класса
rv2 = multivariate_normal([5, 0], [[2**2, 0], [0, 2**2]])  # Построение закона рапсределения 2-го класса

# ИНдикация разности апостериорных распределений классов (взвешанных условных законов) на плоскости
fig = plt.figure(figsize = (8, 6))
ax = fig.add_subplot(111)
Py1 = 0.7  # Априорная вероятность первого класса на входе классификатора
Py2 = 0.3  # Априорная вероятность второго класса на входе классификатора
C = plt.contour(x, y, Py1 * rv1.pdf(pos) - Py2 * rv2.pdf(pos),8,colors='black')
plt.clabel(C, inline=1, fontsize=10)  # индикаци значений правдоподобия

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title(r'$f(\mathbf{x})=' + str(Py1) + 'p_1(\mathbf{x})-' + str(Py2) + 'p_2(\mathbf{x})$')
plt.text(8, 7, s = r'$f(\mathbf{x})>0$', fontsize=12, bbox=dict(color='w'))
plt.text(-2, -3, s = r'$f(\mathbf{x})<0$', fontsize=12, bbox=dict(color='w'))

plt.show()


"""
7. Иллюстрации к принципу работы линейного байесовского классификатора.
Критерий максимального правдоподобия (МП) при разных дисперсиях признаков в классе.
Одинаковые дисперсии у разных классов.

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

# Формирование двумерных условных законов распределений
x, y = np.mgrid[-4:12:.01, -4:9:.1]  # Формирование координатной сетки
pos = np.dstack((x, y))  #Конкатеннация двух одномерных массивов в двумерный
rv1 = multivariate_normal([2, 5], [[3**2, 0], [0, 1.5**2]])  # Построение закона рапсределения 1-го класса
rv2 = multivariate_normal([6, 0], [[3**2, 0], [0, 1.5**2]])  # Построение закона рапсределения 2-го класса

# ИНдикация разности условных распределений классов для критерия МП
fig = plt.figure(figsize = (8, 6))
ax = fig.add_subplot(111)
C = plt.contour(x, y, rv1.pdf(pos)-rv2.pdf(pos),8,colors='black')
plt.clabel(C, inline=1, fontsize=10)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title(r'$f(\mathbf{x})=p_1(\mathbf{x})-p_2(\mathbf{x})$')
plt.text(8, 7, s = r'$f(\mathbf{x})>0$', fontsize=12, bbox=dict(color='w'))
plt.text(-2, -1, s = r'$f(\mathbf{x})<0$', fontsize=12, bbox=dict(color='w'))
plt.show()



"""
8. Иллюстрации к принципу работы линейного байесовского классификатора.
Критерий максимального правдоподобия (МП) при одинаковых дисперсиях признаков в классе.
Разные дисперсии у классов.

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


# Формирование двумерных условных законов распределений
par1 = [2,5,2]  # mx, mym sigma
par2 = [6,0,1.5]  # mx, mym sigma
x, y = np.mgrid[-4:12:.01, -4:9:.1]   # Формирование координатной сетки для графиков
pos = np.dstack((x, y))  # Слияние одномерных массивов в двумерный
# Построение закона рапсределения 1-го класса
rv1 = multivariate_normal([par1[0], par1[1]], [[par1[2]**2, 0], [0, par1[2]**2]])
# Построение закона рапсределения 2-го класса
rv2 = multivariate_normal([par2[0], par2[1]], [[par2[2]**2, 0], [0, par2[2]**2]])

# ИНдикация разности условных распределений классов для критерия МП
fig = plt.figure(figsize = (8, 6))
ax = fig.add_subplot(111)
C = plt.contour(x, y, rv1.pdf(pos)-rv2.pdf(pos),13,colors='black')
plt.clabel(C, inline=1, fontsize=10)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_title(r'$f(\mathbf{x})=p_1(\mathbf{x})-p_2(\mathbf{x})$')
plt.text(8, 7, s = r'$f(\mathbf{x})>0$', fontsize=12, bbox=dict(color='w'))
plt.text(9, 3, s = r'$f(\mathbf{x})<0$', fontsize=12, bbox=dict(color='w'))
plt.text(par1[0]-0.3, par1[1], s = r'$\mathbf{\mu}_1$', fontsize=12, bbox=dict(color='w'))
plt.text(par2[0]-0.3, par2[1], s = r'$\mathbf{\mu}_2$', fontsize=12, bbox=dict(color='w'))
plt.show()



"""
9. Принцип разделения линейно неразделимых классов

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Чтение  и индикаци датасета
df = pd.read_csv('sun_cos_clusters_2x60.csv')
mark = {'y= +1': "s", 'y= - 1': "D"}
sns.scatterplot(data = df, x = "x1", y = "x2",
                hue = "classes", style = "classes", s = 100,
                markers = mark, edgecolors = 'black', linewidths = 2)
plt.plot([0,360], [-1,-1], linewidth = 2, color='black', linestyle = '--')
# plt.text(10, -0.7, s = r'$f(\mathbf{x})=0$', fontsize=12, bbox=dict(color='w'))
plt.show()

# Индикация кластеров и разделяющей плоскости в 3D
X, Y = np.mgrid[-10:360:100j, -4:2:50j]  # Формирование координатной сетки для графиков в 3D
Z = X*Y*0  # Задание нулевой плоскости
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, alpha=0.5)  # Построение разделяющей плоскости
ax.scatter3D(df["x1"], df['x2'], df['sign'] * 1, c = df['sign'], s = 50, edgecolors = 'black')  # Индикация точек

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel('PF')
ax.view_init(20, -120)  # изменение ракурса графика

plt.show()



"""
10. Принцип работы метода потенциальных функция

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Чтение  и индикаци датасета
df = pd.read_csv('sun_cos_clusters_2x60.csv')
mark = {'y= +1': "s", 'y= - 1': "D"}
sns.scatterplot(data=df, x="x1", y="x2",
                hue="classes", style="classes", s=100,
                markers=mark, edgecolors='black', linewidths=2)
plt.plot([0, 360], [-1, -1], linewidth=2, color='black', linestyle='--')
plt.show()

# Задание масштабов потенциальных функций
sig_x1, sig_x2 = 30, 0.2


# Потенциальная функция
def PF_bell(x, y, mx=0.0, my=0, sx=1.0, sy=1.0):
    return np.exp(- (((x - mx) / sx) ** 2 + ((y - my) / sy) ** 2) / 2)


def Cumul_PF(x, y, df, sx, sy):  # Кумулятивное потенциальное поле в точке (x,y)
    sum = 0
    for i in range(len(df)):
        sum += -(df["classNum"][i] - 0.5) * 2 * PF_bell(x, y, df['x1'][i], df['x2'][i], sx, sy)
    return sum


# Задание координатной сетки для графиков
X, Y = np.mgrid[-10:360:100j, -4:2:50j]  # Координатная сетка
# x, y = np.arange(-10, 360, 1), np.arange(-4, 2, 0.1)
# X, Y = np.meshgrid(x, y)

# Кумулятивное потенциальное поле обучающей выборки
Z = Cumul_PF(X, Y, df, sig_x1, sig_x2)

# Значение потенциального рельефа в точках обучающей выборки
Z2 = Cumul_PF(df["x1"], df['x2'], df, sig_x1, sig_x2)

# Индикация потенциального рельефа
fig = plt.figure(figsize=(16, 9))
ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                 cmap='viridis')

ax.plot_wireframe(X, Y, Z, alpha=0.5)  # Индикация проволочного потенциального рельефа
# ax.plot_wireframe(X, Y, Z*0, alpha=0.5)
# ax.scatter3D(df["x1"], df['x2'], df['sign'] * 1, c = df['sign'], s = 50, edgecolors = 'black')
ax.scatter3D(df["x1"], df['x2'], Z2, c=df['sign'], s=50, edgecolors='black')  # Индикация выборки

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel('PF')
ax.view_init(20, -120)  # изменение ракурса графика
plt.show()

# Индикация проволочного потенциального рельефа kbybzvb ehjdyz
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
C = plt.contour(X, Y, Z, 9, colors='black')
plt.clabel(C, inline=1, fontsize=10)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.show()



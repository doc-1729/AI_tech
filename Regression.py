"""
Линейная регрессия
(файл содержит:
1. Восстановление двумерной регрессии с использованием класса LinearRegression() из sklearn.
2. Восстановление двумерной регрессии с помощью стандартных матричных операций.
3. Одномерна регрессия для иллюстраций.
4. Полиномиальная регрессия

"""
import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

rnd.seed(10002)  # Инициалищация генератора случайных чисел

# Исходные данные для генерации синтетической обучающей выборки
W=[2,3,0.1]  # Коэффициенты регрессии [w0, w1, w2]
x_min, x_max = 1, 5  # Интервал варьирования случайных координат x1 и x2
e_sigma = 1  # СКО ско гауссовской случайной составляющей
N = 50  # Размер обучающей выборки

# Функция генерации выборки по заданному уравнению регрессии
def X_2D_generate(rows):  # формирует массив случайных векторов признаков [x1,...,XD] с равномерным законом
    data = []
    for i in range(rows):
        x = [rnd.uniform(x_min, x_max) for j in range(2)]
        data.append([x[0], x[1], W[0] + x[0]*W[1] + x[1]*W[2]+rnd.gauss(0,e_sigma)])
    return pd.DataFrame(data, columns=['x1','x2','y'])


# Генерация синтетической обучающей выборки в датасет стандартного формата DataFrame
df = X_2D_generate(N)
# df.to_csv('x1_x2_y_regresson_50.csv',index=False)  # запись в файл

# Индикация проекции обучающей выборки и исходной функции регрессии на плоскость (x1,y)
sns.scatterplot(data = df, x = "x1", y = "y", s = 100,
                edgecolors = 'black', linewidths = 2)
plt.plot([x_min, x_max], [W[0] + x_min*W[1] + x_min*W[2], W[0] + x_max*W[1] + x_max*W[2]],
         linewidth = 2, color='black', linestyle = '--')
plt.show()

# Извлечение из датасета и форматирование данных для обучения модели линейной регрессии
x = df[['x1','x2']]
x = x.values
y = df['y'].to_numpy()

# Восстановление регрессии
model = LinearRegression().fit(x, y)  # обучает модель и восстанавливает регрессию
r_sq = model.score(x, y)  # коэффициент согласия (детерминации) модели
print('coefficient of determination:', r_sq)  # печать коэффициента согласия модели
print('intercept:', model.intercept_)  # Печать коэффициента w0
print('slope:', model.coef_)  # Печать коэффициентов w1, w2, ...



"""
2. Восстановление двумерной регрессии с помощью стандартных матричных операций.

"""
# Чтение датасета из файла
df = pd.read_csv('x1_x2_y_regresson_50.csv')

# Добавление столбца x0=1 для учета коэф. w0 в матричном решении
df['x0'] = 1

# Извлечение из датасета и форматирование данных для обучения модели линейной регрессии
x = df[['x0','x1','x2']]
x = x.values
y = df['y'].to_numpy()

# Матричное решение
xt = x.transpose()  # Транспонирование матрицы координат X
xx = np.dot(xt,x)  # Произведение матриц: XtX
xx_1 = np.linalg.inv(xx)  # Обращение матрицы XtX
xxx = np.dot(xx_1,xt)  # Произведение матриц : 1/(XtX)*Xt
xxxy =  np.dot(xxx,y)  # Произведение матриц: (1/(XtX)*Xt)*Y
print('[w0,w1,w2]=',xxxy)  # Печать матричного решения



"""
3. Одномерна регрессия

"""
import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

rnd.seed(10005)  # Инициалищация генератора случайных чисел

# Исходные данные для генерации синтетической обучающей выборки
W=[2,3]   # Коэффициенты регрессии [w0, w1]
x_min, x_max = 1, 5  # Интервал варьирования случайных координат x1 и x2
e_sigma = 2  # СКО ско гауссовской случайной составляющей
N = 10  # Размер обучающей выборки

# Функция генерации выборки по заданному уравнению регрессии
def X_1D_generate(rows):  # формирует массив (x,y)
    data = []
    for i in range(rows):
        x = rnd.uniform(x_min, x_max)
        data.append([x, W[0] + x*W[1] + rnd.gauss(0,e_sigma)])
    return pd.DataFrame(data, columns=['x1','y'])


# Генерация синтетической обучающей выборки в датасет стандартного формата DataFrame
df = X_1D_generate(N)

# Извлечение из датасета и форматирование данных для обучения модели линейной регрессии
x = df[['x1']]
x = x.values
y = df['y'].to_numpy()

# Восстановление линейной регрессии
model = LinearRegression().fit(x, y)  # Обучает модель по выборке x,y
print('R^2:', model.score(x, y))  # Печать коэффициента детерминации R**2 модели
ww = np.concatenate (([model.intercept_], model.coef_))  # Восстановленные коэффициенты регрессии ww=[w0,w1]
print('w0:', model.intercept_)  # Печать коэффициента w0 регрессии
print('[w1,...]:', model.coef_)  # Печать коэффициентов [w1, ...] регрессии
print('[w0,w1] = ', ww)

# Индикация обучающей выборки, заданной и восстановленной функций регрессии
sns.scatterplot(data = df, x = "x1", y = "y", s = 100,
                edgecolors = 'black', linewidths = 2)
plt.plot([x_min, x_max], [W[0] + x_min*W[1], W[0] + x_max*W[1]],
         linewidth = 1, color='green', linestyle = '-')
plt.plot([x_min, x_max], [ww[0] + x_min*ww[1], ww[0] + x_max*ww[1]],
         linewidth = 2, color='black', linestyle = '--')

# Прогноз на основе обученной модели
x_new = [[4]]  # Задание интересующего x
y_new = model.predict(x_new)  # Предсказание зависимой величины y по восстановленной модели
print('y(x_new=', x_new[0][0],') =', y_new)

# Индикация предсказанного значения y(x)
plt.scatter([x_new], [y_new], marker = 'o', color = 'red', s = 200)

plt.show()



"""
4. Полиномиальная регрессия для иллюстраций

"""
import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


rnd.seed(10005)  # Инициалищация генератора случайных чисел

# Исходные данные для генерации синтетической обучающей выборки
W=[2,1,-0.5]  # Коэффициенты регрессии y= w0 + w1*x + w2*x**2] + e
x_min, x_max = 1, 5  # Интервал варьирования случайных координат x
e_sigma = 1  # СКО ско гауссовской случайной составляющей
N = 10  # Размер обучающей выборки

# Функция регрессии
def y_x2(x, w:list):
    return (w[0] + w[1]*x + w[2]*x*x)


#  Функция генерации обучающей выборки
def y_x2_generate(rows):  # формирует массив (x,y)
    data = []
    for i in range(rows):
        x = rnd.uniform(x_min, x_max)
        data.append([x, y_x2(x, W) + rnd.gauss(0,e_sigma)])
    return pd.DataFrame(data, columns=['x1','y'])

# Генерация синтетической обучающей выборки в датасет стандартного формата DataFrame
df = y_x2_generate(N)

# Извлечение из датасета и форматирование данных для обучения модели линейной регрессии
x = df[['x1']]
x = x.values
y = df['y'].to_numpy()

# Восстановление регрессии
pf=PolynomialFeatures(degree=2)  # Формирует объект полимодели
X_poly=pf.fit_transform(x)  # Формирует три столбца (1, x, x^2) из столбца X
# print(X_poly)
lm2 = LinearRegression().fit(X_poly, y)  # Обучение модели линейной регрессии

# Печать параметров восстановленной модели регрессии
print('coefficient of determination:', lm2.score(X_poly, y))  # печать коэффициента детерминации модели
print('w0:', lm2.intercept_)  # Печать w0
print('(w1,w2,...):', lm2.coef_)  # Печать w1, w2, ...


# Индикация обучающей выборки, заданной и восстановленной функций регрессии
sns.scatterplot(data = df, x = "x1", y = "y", s = 100,
                edgecolors = 'black', linewidths = 2)

xx = np.arange(x_min,x_max,0.1)  # Ммассив координат x для построения линии регрессии
xx = xx.reshape(len(xx),1)  # Перформатирование в столбец элементов
plt.plot(xx, y_x2(xx, W),
         linewidth = 1, color='green', linestyle = '-')  # Заданная линия регрессии
plt.plot(xx, lm2.predict(pf.fit_transform(xx)),
         linewidth = 2, color='black', linestyle = '--')  # Восстановленная линия регрессии

# Прогноз на основе обученной модели
x_new = [[4]]  # Задание интересующего x
y_new = lm2.predict(pf.fit_transform(x_new))  # Предсказание зависимой величины y по восстановленной модели
print('y(', x_new[0][0],')=', y_new)

# Индикация предсказанного значения y(x) на графике
plt.scatter([x_new], [y_new], marker = 'o', color = 'red', s = 200)  # наблюдаемый образ

plt.show()


# Украшение графика сеткой и легендой
fig, ax = plt.subplots()
ax.set(xlabel = '$x$')
ax.set_ylabel('$y$')
plt.scatter(x, y,s = 60)
plt.plot(xx, y_x2(xx, W),
         linewidth = 1, color='green', linestyle = '-')
plt.plot(xx, lm2.predict(pf.fit_transform(xx)),
         linewidth = 2, color='black', linestyle = '--')
x_new = [[4]]
y_new = lm2.predict(pf.fit_transform(x_new))
plt.scatter([x_new], [y_new], marker = 'o', color = 'red', s = 150)
plt.legend([r'{$(x_i,y_i)$}','$y(x)$','$a(x)$','a(x=4)'])
ax.grid()

plt.show()
"""
Логистическая регрессия. Метод Ньютона-Рафсона
2 класса, 2D признаковое пространство.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rnd
import seaborn as sns

rnd.seed(10003)  # Инициализация генератора сучайных чисел


# Функция генерация кластеров
def norm_2D_classes_generate(rows):
    m_s_xy = [[8, 2, 3, 2], [3, 2, 6, 2]]  # в каждой строке M_x,Sig_x, M_y, Sig_y
    n_classes = len(m_s_xy)
    data = []
    for classNum in range(n_classes):
        for i in range(rows):
            data.append([rnd.gauss(m_s_xy[classNum][0], m_s_xy[classNum][1]),
                         rnd.gauss(m_s_xy[classNum][2], m_s_xy[classNum][3]), classNum, 2 * classNum - 1])
    return pd.DataFrame(data, columns=['x1', 'x2', 'class', 'y'])


# Генерация датасета в формате DataFrame
df = norm_2D_classes_generate(60)
# df.to_csv('x1_x2_class_with_None_2x60.csv',index=False)  # запись в файл
X = df[['x1', 'x2']]  # Извлечение из датачета матрицы объектов X
y = df['class']  # и Извлечение из датасета вектора меток класса y

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Разделение размеченной выборки на train и test части
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Обучение логит-модели классификатора
logreg = LogisticRegression()  # Инициализация модели обучения
gx1x2 = logreg.fit(X_train, y_train)  # Обучение модели

# Печать результатов обучения
W = [gx1x2.intercept_[0], gx1x2.coef_[0][0], gx1x2.coef_[0][1]]  # Извлечение найденных параметров модели
print('Параметры модели классификатора логистической регрессии:')
print('(w0, w1, w2) = ', W)
print('n_iter:', gx1x2.n_iter_[0])  # Число итераций обучения

# индикация кластеров
x1 = df.loc[df['class'] == 0].x1
x2 = df.loc[df['class'] == 0].x2
fig, ax = plt.subplots()
plt.scatter(x1, x2, marker='s', color='yellow', edgecolors='black', s=50)
x1 = df.loc[df['class'] == 1].x1
x2 = df.loc[df['class'] == 1].x2
plt.scatter(x1, x2, marker='D', color='skyblue', edgecolors='black', s=70)
plt.legend(['y=0', 'y=1'])
ax.set_xlabel('$x1$')
ax.set_ylabel('$x2$')
ax.grid()


# Индикация графика разделяющей функци классификатора (разделяющей линии)
def x2_from_x1(x1, W):
    return (- W[0] - W[1] * x1) / W[2]


x_min, x_max = 2.3, 10
plt.plot([x_min, x_max], [x2_from_x1(x_min, W), x2_from_x1(x_max, W)],
         linewidth=2, color='black', linestyle='-', alpha=0.7)
plt.text(1.8, 0, s=r'$g(x,w)=0$', fontsize=12, bbox=dict(color='w'), rotation=0)
plt.show()

# Kлассификация объектов тестовой выборки
y_pred = logreg.predict(X_test)
accuracy = logreg.score(X_test, y_test)  # Метрика accuracy классификатора
print('\n Точность (accuracy) классификатора логистической регрессии на тестовом наборе: {:.2f}'.format(accuracy))

# Печать матрицы ошибок для тестовой части набора данных
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print('\n Матрица ошибок: ')
print(cnf_matrix)

# Печать основных показателей качества классификации
from sklearn.metrics import classification_report

print('\n Отчет по метрикам классификации')
print(classification_report(y_test, y_pred))

# рабочие характеристики классификатора (ROC)
y_pred_test = logreg.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_test)
auc = metrics.roc_auc_score(y_test, y_pred_test)

# Индикация рабочих характеристик бинарного классификатора (ROC-кривой)
fig, ax = plt.subplots()
plt.plot(fpr, tpr, label='Logistic Regression (AUC = %0.3f)' % auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
ax.grid()

plt.show()

# Печать интегральных показателей модели классификатора из другой библиотеки (statsmodels.Logi)
y = df['class']
import statsmodels.api as sm

logit_model = sm.Logit(y, X)  # Инициализация логит модели
result = logit_model.fit()  # Обучение модели классификатора
print('\n Интегральные показатели обученной модели классификатора из другой библиотеки (statsmodels.Logi):')
print('\n 1. result.summary2:')
print(result.summary2())  #
print('\n 2. result.summary:')
print(result.summary())
print('\n 3. result.params:')
print(result.params)
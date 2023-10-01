"""
Обучается нейросеть ставить диагноз: M = malignant (злокачественная) / B = benign (доброкачественная)), -
относительно описанию ренгеновского снимка опухоли молочной железы.
1. Подготовка данных.
2. Создание и обучение нейросети
3. Тестирование нейросети
Набор данных: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
Рекомендуется выполнять в Colab: https://colab.research.google.com/  в режиме с GPU-ускорителем

"""

import tensorflow as tf
import keras as ks
import numpy as np
import pandas as pd

"""
1. Подготовка данных.

"""
# Чтение датасета из файла в Dataframe
numeric_column_names = ['x'+str(i+1) for i in range(30)]  # Задаем имена колонкам признаков
# from google.colab import files
# file = files.upload()
df = pd.read_csv('wdbc.data',
                 names=['ID','Diag'] + numeric_column_names)

# Удаление столбца с ID
df.drop(columns = ['ID'], axis = 1, inplace=True)

# Нормализация данных. Смещение - по медиане, сжатие по СКО
df[numeric_column_names] = (df[numeric_column_names]-df[numeric_column_names].median())/df[numeric_column_names].std()

# Замена категориальных ответов на числовые - добавляем столбец числовых ответов
# df['Y'] = pd.factorize(df['Diag'])[0]*2 -1  # Добавление столбца в конец
df.insert(loc= 0 , column='Y', value=-pd.factorize(df['Diag'])[0]+1)  # Вставка столбца в заданное место

# # Разделение имеющихся данных на обучающую и тестовую выборку в заданной пропорции
X_df = df[numeric_column_names]
Y_df = df['Y']
from sklearn. model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y_df, test_size=0.2, random_state=1)


"""
2. Создание и обучение нейросети

"""
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

tf.test.gpu_device_name()  # Проверка доступности GPU в среде исполнения.
# В случае успешности будет выдан примерно такой результат '/device:GPU:0'

classifier = Sequential()  # Инициализация нейросети
classifier.add(Dense(units = 16, activation = 'relu', input_dim = 30))
classifier.add(Dense(units = 8, activation = 'relu'))
classifier.add(Dense(units = 6, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Выбор метода оптимизации и функционала качества
classifier.compile(optimizer = 'Nadam', loss = 'binary_crossentropy')

# Обучение нейросети
classifier.fit(X_train, Y_train, batch_size = 1, epochs = 25)


"""
3. Тестирование нейросети

"""
Y_pred = classifier.predict(X_test) # Обученная НС дает ответы на тестовый набор данных
Y_pred = [ 1 if y>=0.5 else 0 for y in Y_pred ]  # Бинаризация вероятности прогноза

from sklearn. model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# Печать матрицы ошибок для тестовой части набора данных
cnf_matrix = metrics.confusion_matrix (Y_test, Y_pred)
print('\n Матрица ошибок: ')
print(cnf_matrix)

# Печать основных показателей качества классификации
from sklearn.metrics import classification_report
print('\n Отчет по метрикам классификации')
print(classification_report(Y_test, Y_pred))
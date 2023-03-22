import numpy as np
import random as rnd
import matplotlib.pyplot as plt

rnd.seed(10003)

# Решение задачи коммивояжера с помощью генетического алгоритма

# Исходные данные
V0 = [0, 1, 2, 3, 4]  # Нулевая хромосома (маршрут)
E = [[0, 5, 4, 6, 1],  # Матрица расстояний между городами
     [5, 0, 4, 2, 1],
     [4, 4, 0, 1, 2],
     [6, 2, 1, 0, 3],
     [1, 1, 2, 3, 0]]
K = 7  # Размер популяции
M_child = 2  # Число потомков в каждом поколении
k_cross = 2  # Точка разрыва хромосомы при скрещивании
Pmut = 0.8  # Вероятность мутации
Nmax = 10  # Максимальное число поколений


# Функция генерации массива из K строк из перемешанного V0
def Pop_generation(V0, K):
    n = len(V0)
    V0.append(-1)
    B = [V0]
    for i in range(K - 1):
        B.append([0] + sorted(V0[1:n], key=lambda A: rnd.randrange(1000)) + [-1])
    return B


# Фитнес-функция маршрута
def fitnes(Vk, E):
    n = len(E)
    sum = 0
    for i in range(n):
        sum += E[Vk[i]][Vk[(i + 1) % n]]
    return sum


# Функция одноточечного скрещивания для одного потомка
def crossover_1(A, B):
    C = A[:2] + B[2:]
    D = B[2:-1]  # Перекрещиваем хромосомы для первого потомка
    if C[1] in D:
        k = B.index(C[1])  # Запоминаем индекс повторного гена в правой части хромосомы
        for i in range(2, len(A) - 1):
            if not (A[i] in D):
                C[k] = A[i]
                break
    C[-1] = fitnes(C, E)
    return C


# Функция мутации перестановкой случйной пары генов
def mutation(A, Pmut):
    if rnd.random() < Pmut:
        poz = np.random.randint(1, len(A) - 1, 2)
        A[poz[0]], A[poz[1]] = A[poz[1]], A[poz[0]]
    A[-1] = fitnes(A, E)
    return A


# Генерация начальной популяции
V = Pop_generation(V0, K)

# Вычисление фитнес-функции популяции (длины марщрута)
for i in range(len(V)):
    V[i][-1] = fitnes(V[i], E)

# Выбор пары родителей
V.sort(key=lambda x: x[5])  # Сортировка популяции по возрастанию длины маршрута


# Функция цикла по эпохам (поколениям) генетического алгоритма
def evolution(V):
    F = []  # Массив для запоминания лучшей фитнес-функции каждого поколения
    for i in range(Nmax):  # Цикл по поколениям
        # Скрещивание
        V_child = [crossover_1(V[0], V[1]),
                   crossover_1(V[1], V[0])]  # Массив потомков после скрещивания
        # Мутация потомков
        V_mut = [mutation(V_child[0], Pmut), mutation(V_child[1], Pmut)]  # Мутация потомков
        # Расширение популяции потомками
        V = V + V_mut
        # Сортировка популяции по возрастанию длины маршрута
        V.sort(key=lambda x: x[-1])
        # Сокращение популяции
        del V[K:]
        # Для индикации результатов
        # print('Pop',i+1,':',V)  # Печать популяции поколения
        F.append(V[0][-1])  # Запоминание лучшей фитнес-функции поколения
    print('C=', F)  # Для проверки графика лучших особей фитнес-функций поколений
    return F  # Возвращает массив лучших фитнес-функций поколений


# evolution(V)  # однократный запуск генетического алгоритма


# Набор статистики результатов генетического алгоритма
Ncicle = 30  # Задаем число циклов всего генетического алогритма (графиков сходимости)
# Формируем в цикле массив хромосом лучших особей гаждого поколения каждого повтора генетического алгоритма
FF = []
for i in range(Ncicle):
    FF.append(evolution(V))  # Запуск генетического алгоритма

# Построение графика наилучшей особи от поколения
fig, ax = plt.subplots(figsize=(4, 2.5))
xlist = [i for i in range(1, Nmax + 1)]  # Формируем ось абсцисс графика - число поколений
ax.set_xlabel('поколение')
ax.set_ylabel('$C(v,E)$')
for i in range(Ncicle):
    plt.plot(xlist, FF[i])
plt.show()
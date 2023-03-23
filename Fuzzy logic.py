# Нечеткая логика

import matplotlib.pyplot as plt
import numpy as np

# Примеры функций принадлежности

# Кусочно-линейные функции принадлежности
Xmin, Xmax = 0.0, 100.0  # Границы универсума
p_x = [20, 40, 60, 80]  # Точки перегиба. Массив д.б. по возрастанию.

# Функции принадлежности
# Скошенный фронт
def MF_01(x, a, b):
    if a <= x <= b:
        return (x - a) / (b - a)
    else:
        return 0 if x < a else 1


# Скошенный срез
def MF_10(x, a, b):
    if a <= x <= b:
        return 1 - (x - a) / (b - a)
    else:
        return 1 if x < a else 0


# Трапеция
def MF_0110(x, a, b, c, d):
    if x < a or x > d:
        return 0
    elif a <= x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return 1
    else:
        return 1 - (x - c) / (d - c)


# Трегольник
def MF_010(x, a, b, c):
    if x < a or x > c:
        return 0
    elif a <= x <= b:
        return (x - a) / (b - a)
    else:
        return 1 - (x - b) / (c - b)


xlist = np.linspace(Xmin, Xmax, 200)
ylist = np.asarray([MF_01(x, p_x[2], p_x[3]) for x in xlist])

fig, ax = plt.subplots(figsize=(4, 2.5))
ax.set_xlabel('$x$')
ax.set_ylabel('$\mu (x)$')
plt.xlim(Xmin, Xmax)

plt.plot(xlist, ylist,
         linewidth=3, color='orange', linestyle='--', alpha=0.7)

ylist = np.asarray([MF_10(x, p_x[0], p_x[1]) for x in xlist])
plt.plot(xlist, ylist,
         linewidth=3, color='green', linestyle='-.', alpha=0.7)

ylist = np.asarray([MF_0110(x, p_x[0], p_x[1], p_x[2], p_x[3]) for x in xlist])
plt.plot(xlist, ylist,
         linewidth=3, color='blue', linestyle='-', alpha=0.7)

# ylist = np.asarray([MF_010(x, p_x[0], p_x[1], p_x[2]) for x in xlist])
# plt.plot(xlist, ylist,
#         linewidth = 3, color='blue', linestyle = '-', alpha=0.7)


plt.text(3, 0.6, s="мало", fontsize=12, bbox=dict(color='w'), rotation=0)
plt.text(40, 0.5, s="норма", fontsize=12, bbox=dict(color='w'), rotation=0)
plt.text(78, 0.6, s="много", fontsize=12, bbox=dict(color='w'), rotation=0)
ax.grid()
plt.show()


# Гауссоида
Xmin, Xmax = 0.0, 100.0  # Границы универсума
p_x = [20, 40, 60, 80]  # Точки перегиба. Массив по возрастанию.

xlist = np.linspace(Xmin, Xmax, 200)
ylist = np.exp(-(xlist-60)**2/100)
fig, ax = plt.subplots(figsize = (4,2.5))
ax.set_xlabel('$x$')
ax.set_ylabel('$\mu (x)$')
plt.xlim(Xmin, Xmax)
plt.plot(xlist, ylist,
        linewidth = 3, color='blue', linestyle = '-', alpha=0.7)
plt.text(50, 0.5, s = "$x=60$", fontsize=12, bbox=dict(color='w'), rotation=0)
ax.grid()
plt.show()


# Сигмоиды
Xmin, Xmax = 0.0, 100.0  # Границы универсума
p_x = [20, 40, 60, 80]  # Точки перегиба. Массив по возрастанию.
xlist = np.linspace(Xmin, Xmax, 200)
ylist = 1 / (1 + np.exp(-(xlist - 80)/2))
fig, ax = plt.subplots(figsize = (4,2.5))
ax.set_xlabel('$x$')
ax.set_ylabel('$\mu (x)$')
plt.xlim(Xmin, Xmax)
plt.plot(xlist, ylist,
        linewidth = 3, color='blue', linestyle = '-', alpha=0.7)
ylist = 1 - 1 / (1 + np.exp(-(xlist - 20)/2))
plt.plot(xlist, ylist,
        linewidth = 3, color='blue', linestyle = '--', alpha=0.7)
ax.grid()
plt.show()


#
# Регулятор климатической установки на нечеткой логике
#
fig, axes = plt.subplots(nrows=4, ncols=3, figsize = (10, 7.5), constrained_layout=True)

# Лингвистическая переменная "температура в салоне автомобиля"
tmin, tmax = 0.0, 40.0  # Диапазон влажности
p_t = [15, 22, 27]
tlist = np.linspace(tmin, tmax, 200)
MF11 = lambda x: 1 / (1 + np.exp(-(x - 25)/2))
hot = MF11(tlist)
MF21 = lambda x: np.exp(-(x-22)**2/50)
worm = MF21(tlist)
MF31 = lambda x: 1 - 1 / (1 + np.exp(-(x - 15)/2))
cold = MF31(tlist)

axes[0,0].plot(tlist, hot, linewidth = 3, color='red', linestyle = '-', alpha=0.7)
axes[0,0].set(title = 'жарко')
axes[0,0].text(2, 0.6, s = r'$\mu_{11}(x_1)$', fontsize=12, bbox=dict(color='w'), rotation=0)
axes[1,0].plot(tlist, worm, linewidth = 3, color='green', linestyle = '-', alpha=0.7)
axes[1,0].set(title = 'тепло')
axes[1,0].text(2, 0.6, s = r'$\mu_{21}(x_1)$', fontsize=12, bbox=dict(color='w'), rotation=0)
axes[2,0].plot(tlist, cold, linewidth = 3, color='blue', linestyle = '-', alpha=0.7)
axes[2,0].set(title = 'холодно')
axes[2,0].text(2, 0.6, s = r'$\mu_{31}(x_1)$', fontsize=12, bbox=dict(color='w'), rotation=0)


# Лингвистическая переменная "влажность"
hmin, hmax = 0.0, 100.0  # Диапазон влажности
p_h = [30, 70]  # Точки функций принадлежности

hlist = np.linspace(hmin, hmax, 200)
MF12 = lambda x: 1 / (1 + np.exp(-(x - p_h[1])/2))
dump = MF12(hlist)
MF22 = lambda x: 1 / (1 + np.exp(-(x - p_h[0])/2)) - 1 / (1 + np.exp(-(x - p_h[1])/2))
norm = MF22(hlist)
MF32 = lambda x: 1 - 1 / (1 + np.exp(-(x - p_h[0])/2))
dry = MF32(hlist)

axes[0,1].plot(hlist, dump, linewidth = 3, color='blue', linestyle = '-', alpha=0.7)
axes[0,1].set(title = 'сыро')
axes[0,1].text(2, 0.6, s = r'$\mu_{12}(x_2)$', fontsize=12, bbox=dict(color='w'), rotation=0)
axes[1,1].plot(hlist, norm, linewidth = 3, color='green', linestyle = '-', alpha=0.7)
axes[1,1].set(title = 'норма')
axes[1,1].text(2, 0.6, s = r'$\mu_{22}(x_2)$', fontsize=12, bbox=dict(color='w'), rotation=0)
axes[2,1].plot(hlist, dry, linewidth = 3, color='orange', linestyle = '-', alpha=0.7)
axes[2,1].set(title = 'сухо')
axes[2,1].text(2, 0.6, s = r'$\mu_{32}(x_2)$', fontsize=12, bbox=dict(color='w'), rotation=0)


# Лингвистическая переменная "режим климатической установки"
cmin, cmax = -20.0, 40.0  # Диапазон влажности
p_c = [-10, 22, 30]  # Точки функций принадлежности

clist = np.linspace(cmin, cmax, 200)
MF13 = lambda x: 1 - 1 / (1 + np.exp(-(x - p_c[0])/2))
ccold = MF13(clist)
MF23 = lambda x: 1 / (1 + np.exp(-(x - p_c[0])/2)) - 1 / (1 + np.exp(-(x - p_c[1])/2))
cok = MF23(clist)
MF33 = lambda x: 1 / (1 + np.exp(-(x - p_c[1])/2))
cworm = MF33(clist)

axes[0,2].plot(clist, ccold, linewidth = 3, color='blue', linestyle = '-', alpha=0.7)
axes[0,2].set(title = 'морозить')
axes[0,2].text(2, 0.6, s = r'$\mu_{1}(y)$', fontsize=12, bbox=dict(color='w'), rotation=0)
axes[1,2].plot(clist, cok, linewidth = 3, color='green', linestyle = '-', alpha=0.7)
axes[1,2].set(title = 'выключить')
axes[1,2].text(2, 0.6, s = r'$\mu_{2}(y)$', fontsize=12, bbox=dict(color='w'), rotation=0)
axes[2,2].plot(clist, cworm, linewidth = 3, color='red', linestyle = '-', alpha=0.7)
axes[2,2].set(title = 'греть')
axes[2,2].text(2, 0.6, s = r'$\mu_{3}(y)$', fontsize=12, bbox=dict(color='w'), rotation=0)


# Фазификация
t, h = 20, 71.0

# Правило 1
M_t = MF11(t)
axes[0,0].plot([t, tmax], [M_t, M_t], linewidth = 2, color='black', linestyle = '--', alpha=0.7)
axes[0,0].plot([t, t], [0, 1], linewidth = 2, color='black', linestyle = ':', alpha=0.7)
M_h = MF12(h)
axes[0,1].plot([h, hmax], [M_h, M_h], linewidth = 2, color='black', linestyle = '--', alpha=0.7)
axes[0,1].plot([h, h], [0, 1], linewidth = 2, color='black', linestyle = ':', alpha=0.7)
u1 = max(M_t,M_h)  # Жарко ИЛИ Сыро
ccold_ = [min(u1,x) for x in ccold]
axes[0,2].plot(clist, ccold_, linewidth = 2, color='gray', linestyle = '-', alpha=0.7)
axes[0,2].fill_between(clist, ccold_, alpha = 0.4)

# Правило 2
M_t = MF21(t)
axes[1,0].plot([t, tmax], [M_t, M_t], linewidth = 2, color='black', linestyle = '--', alpha=0.7)
axes[1,0].plot([t, t], [0, 1], linewidth = 2, color='black', linestyle = ':', alpha=0.7)
M_h = MF22(h)
axes[1,1].plot([h, hmax], [M_h, M_h], linewidth = 2, color='black', linestyle = '--', alpha=0.7)
axes[1,1].plot([h, h], [0, 1], linewidth = 2, color='black', linestyle = ':', alpha=0.7)
u2 = min(M_t, M_h)  # Тепло И Норма
cok_ = [min(u2,x) for x in cok]
axes[1,2].plot(clist, cok_, linewidth = 2, color='gray', linestyle = '-', alpha=0.7)
axes[1,2].fill_between(clist, cok_, alpha = 0.4)

# Правило 3
M_t = MF31(t)
axes[2,0].plot([t, tmax], [M_t, M_t], linewidth = 2, color='black', linestyle = '--', alpha=0.7)
axes[2,0].plot([t, t], [0, 1], linewidth = 2, color='black', linestyle = ':', alpha=0.7)
M_h = MF32(h)
axes[2,1].plot([h, hmax], [M_h, M_h], linewidth = 2, color='black', linestyle = '--', alpha=0.7)
axes[2,1].plot([h, h], [0, 1], linewidth = 2, color='black', linestyle = ':', alpha=0.7)
u3 = min(M_t, M_h)  # Тепло И Норма
cworm_ = [min(u3,x) for x in cworm]
axes[2,2].plot(clist, cworm_, linewidth = 2, color='gray', linestyle = '-', alpha=0.7)
axes[2,2].fill_between(clist, cworm_, alpha = 0.4)

# Агрегация
agr = [max(ccold_[i], cok_[i], cworm_[i]) for i in range(len(clist))]
axes[3,2].plot(clist, agr, linewidth = 2, color='black', linestyle = '-', alpha=0.7)
axes[3,2].fill_between(clist, agr, alpha = 0.4)
# Находим центр тяжести
m_center = np.sum(np.asarray([clist[i] * agr[i] for i in range(len(clist))])/np.sum(agr))

axes[3,2].plot([m_center, m_center], [0, 1], linewidth = 2, color='black', linestyle = ':', alpha=0.7)
axes[3,2].text(15, 0.75, s = r'$y*=$' + str(m_center)[:4], fontsize=12, bbox=dict(color='w'), rotation=0)
axes[3,2].text(5, 0.6, s = r'$\mu(y)$', fontsize=12, bbox=dict(color='w'), rotation=0)

axes[3,0].remove()
axes[3,1].remove()
plt.show()
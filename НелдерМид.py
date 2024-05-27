import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x[0]**2 + 2*x[1]**2 - x[1]*3

def nelder_mead(f, x_start,
                step=0.1, no_improve_thr=10e-6,
                no_improv_break=10, max_iter=0,
                alpha=1., gamma=2., rho=-0.5, sigma=0.5):
    # Инициализация симплекса
    dim = len(x_start)
    prev_best = f(x_start)
    no_improv = 0
    res = [[x_start, prev_best]]
    for i in range(dim):
        x = x_start.copy()
        x[i] = x[i] + step
        score = f(x)
        res.append([x, score])

    # Сохранение траектории поиска
    trajectory = [x_start.copy()]

    # Сортировка списка по значениям функции
    iters = 0
    while True:
        # Сортировка симплекса
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # Проверка на завершение
        if max_iter and iters >= max_iter:
            return res[0], trajectory
        iters += 1

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0], trajectory, iters

        # Рефлексия
        x0 = np.mean([x for x, _ in res[:-1]], axis=0)
        xr = x0 + alpha * (x0 - res[-1][0])
        rscore = f(xr)
        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            trajectory.append(xr.copy())
            continue

        # Экспансия
        if rscore < res[0][1]:
            xe = x0 + gamma * (xr - x0)
            escore = f(xe)
            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                trajectory.append(xe.copy())
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                trajectory.append(xr.copy())
                continue

        # Контракция
        xc = x0 + rho * (res[-1][0] - x0)
        cscore = f(xc)
        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            trajectory.append(xc.copy())
            continue

        # Редукция
        x1 = res[0][0]
        nres = []
        for x, _ in res:
            redx = x1 + sigma * (x - x1)
            score = f(redx)
            nres.append([redx, score])
            trajectory.append(redx.copy())
        res = nres

# Начальная точка
x_start = np.array([6.0, -7.0])
result, trajectory, iters = nelder_mead(f, x_start)

print("Минимум найден в точке:", result[0])
print("Значение функции в минимуме:", result[1])
print("Число итераций", iters)

# Построение линий уровня
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x, y)
W = f([X, Y])

plt.contour(X, Y, W, levels=50)
plt.scatter(result[0][0], result[0][1], color='r', marker='o')
plt.plot(*zip(*trajectory), marker='o', color='r')
plt.title('Двумерные линии уровня и траектория поиска')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
# Функция
def f(x):
    x1, x2 = x
    return x1 * np.exp(7/2) * np.exp(x1/2) + x2**2 * np.exp(7/2) * np.exp(x1/2) - 8 * x2 * np.exp(7/2) * np.exp(x1/2) + 23 * np.exp(7/2) * np.exp(x1/2)

# Градиент функции
def grad_f(x):
    x1, x2 = x
    df_dx1 = np.exp(7/2) * np.exp(x1/2) * (1 + x1/2) + x2**2 * np.exp(7/2) * (1/2) * np.exp(x1/2) - 8 * x2 * np.exp(7/2) * (1/2) * np.exp(x1/2) + 23 * np.exp(7/2) * (1/2) * np.exp(x1/2)
    df_dx2 = 2 * x2 * np.exp(7/2) * np.exp(x1/2) - 8 * np.exp(7/2) * np.exp(x1/2)
    return np.array([df_dx1, df_dx2])

# Гессиан функции
def hess_f(x):
    x1, x2 = x
    d2f_dx1dx1 = np.exp(7/2) * np.exp(x1/2) * (1/4) + x2**2 * np.exp(7/2) * (1/4) * np.exp(x1/2) - 8 * x2 * np.exp(7/2) * (1/4) * np.exp(x1/2) + 23 * np.exp(7/2) * (1/4) * np.exp(x1/2)
    d2f_dx1dx2 = x2 * np.exp(7/2) * np.exp(x1/2)
    d2f_dx2dx1 = d2f_dx1dx2
    d2f_dx2dx2 = 2 * np.exp(7/2) * np.exp(x1/2)
    return np.array([[d2f_dx1dx1, d2f_dx1dx2], [d2f_dx2dx1, d2f_dx2dx2]])

# Метод Ньютона-Рафсона
def newton_raphson(f, grad_f, hess_f, x0, tol=1e-5 ):
    x = x0
    k=0
    points = [x0]
    while True:
        grad = grad_f(x)
        hess = hess_f(x)
        x_new = x - np.linalg.inv(hess).dot(grad)
        points.append(x_new)
        if np.linalg.norm(grad) < tol:
            break
        x = x_new
        k+=1
    return x,k,points 

x0 = np.array([8.0, -7.0])

# Поиск минимума
minimum, k, points = newton_raphson(f, grad_f, hess_f, x0)

# Создание сетки значений для линий уровня
x1_vals = np.linspace(-40, 20, 100)
x2_vals = np.linspace(-100, 100, 100)
X1, X2 = np.meshgrid(x1_vals, x2_vals)
Z = f([X1, X2])

# Построение линий уровня
plt.figure(figsize=(8, 6))
contours = plt.contour(X1, X2, Z, levels=np.logspace(0, 5, 35), cmap='jet')
plt.clabel(contours, inline=True, fontsize=8)

# Построение траектории поиска
points = np.array(points)
plt.plot(points[:, 0], points[:, 1], 'ro-') # точки красным цветом с линиями
plt.plot(minimum[0], minimum[1], 'bo') # минимум синим цветом

# Настройки графика
plt.title('Траектория поиска минимума и линии уровня функции')
plt.xlabel('x1')
plt.ylabel('x2')
plt.colorbar(contours)
plt.grid(True)
plt.show()
print(minimum,f(minimum),k)
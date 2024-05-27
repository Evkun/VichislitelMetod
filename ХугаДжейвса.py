import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x[0]**2 + 2*x[1]**2 - x[1]*3

def hooke_jeeves(f, x0, tol=1e-1, step_size=0.5):
    x = x0
    iteration = 0
    delta = np.array([step_size, step_size])
    trajectory = [x.copy()] 
    while np.max(delta) > tol:
        f_min = f(x)
        x_star = x.copy()
        iteration += 1
        for i in range(len(x)):
            for sign in [-1, 1]:
                x_new = x + sign * delta * np.eye(len(x))[i]
                f_new = f(x_new)
                if f_new < f_min:
                    f_min = f_new
                    x_star = x_new
        if f_min < f(x):
            x = x_star
            trajectory.append(x.copy())  
        else:
            delta *= 0.5
    return x, iteration, trajectory 

x0 = np.array([6.0, -7.0])
result, k, trajectory = hooke_jeeves(f, x0)  
print("Минимум найден в точке:", result)

# Построение линий уровня
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
X, Y = np.meshgrid(x, y)
W = f([X, Y])

plt.contour(X, Y, W, levels=50)
plt.scatter(result[0], result[1], color='r', marker='o')
plt.title('Двумерные линии уровня и траектория поиска')
plt.xlabel('x')
plt.ylabel('y')

# Отображение траектории поиска
trajectory = np.array(trajectory)
plt.plot(trajectory[:, 0], trajectory[:, 1], 'bo-')

plt.show()
print("Количество итераций:", k)
print("Значение функции в минимуме:", f(result))

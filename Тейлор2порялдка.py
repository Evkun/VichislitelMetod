import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from numpy.linalg import norm

x=[0,2]
init_cond = [1,0] #y(0)=1, y'(0)=0


#ФУНКЦИИ

def f(x):
    return x*np.exp(-x)

def diff_system(args, x):
    y1, y2 = args
    return [y2, x * np.exp(-x) - y1]

def taylor(count, x_limits, initial):
    xmin, xmax = x_limits
    y1, y2 = initial
    h = (xmax - xmin) / (2 ** count)
    x = xmin
    N = 0
    X = np.array([x])
    Y1 = np.array([y1])
    Y2 = np.array([y2])

    while x < xmax:
        x += h
        N += 1

        k11 = h * y2
        k12 = h * (x * np.exp(-x) - y1)
        k21 = h * (y2 + k12 / 2)
        k22 = h * (x * np.exp(-x) - (y1 + k11 / 2))

        y1_next = y1 + k21
        y2_next = y2 + k22

        y1, y2 = y1_next, y2_next

        X = np.append(X, x)
        Y1 = np.append(Y1, y1)
        Y2 = np.append(Y2, y2)

    return [X, Y1, Y2, h]

#ПОИСК РЕШЕНИЯ


eps = 0.01
N = 0

while True:
    T = taylor(N, x, init_cond)
    T2 = taylor(N+1, x, init_cond)
    error = (norm(T[1] - T2[1][0::2]) + norm(T[2] - T2[2][0::2]))/(norm(T2[1]) + norm(T2[2]))
    N += 1
    if error < eps:
        #print(error)
        #print(N)
        break
X, Y, P, h = T2

#Встроенная функция
sol = odeint(diff_system, init_cond, X)
Y_libr = sol[:,0]
P_libr = sol[:,1]


#y(x)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(X,Y, label = "Численное")
plt.plot(X,Y_libr, label = "Библитечное")
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper right')
plt.title('График решения y(x)')
plt.subplot(1,2,2)
plt.plot(X,abs(Y - Y_libr))
plt.xlabel('x')
plt.title('Разностный график')
plt.show()

#y'(x)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(X,P, label = "Численное")
plt.plot(X,P_libr, label = "Библитечное")
plt.xlabel('x')
plt.ylabel("y'")
plt.legend(loc='upper right')
plt.title("График решения y'(x)")
plt.subplot(1,2,2)
plt.plot(X,abs(P - P_libr))
plt.xlabel('x')
plt.title('Разностный график')
plt.show()

#y'(y)
plt.plot(Y,P, label = "Численное")
plt.plot(Y_libr,P_libr, label = "Библитечное")
plt.xlabel("y")
plt.ylabel("y'")
plt.title("Фазовая траектория y'(y)")
plt.show()
import numpy as np

def fx(x, y):
  return x**3 + y**3 -8

def fy(x, y):
  return y-3+x**(3/2)

def dfx(x, y):
  return 3*x**2 +3* y**2 

def dfy(x, y):
  return 1+(3/2)*x**(1/2)
x0 = 1.0  # начальное приближение для x
y0 = 1.0  # начальное приближение для y
eps = 1e-4  # точность решения

x1 = x0 - fx(x0, y0) / dfx(x0,y0)
y1 = y0 - fy(x0, y0) / dfy(x0,y0)
while (fx(x0, y0)**2 + fy(x0, y0)**2) > eps:
    x0 = x1
    y0 = y1
    x1 = x0 - fx(x0, y0) / dfx(x0,y0)
    y1 = y0 - fy(x0, y0) / dfy(x0,y0)
    x0 = x1
    y0 = y1
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

print('x =', x0, 'y =', y0)

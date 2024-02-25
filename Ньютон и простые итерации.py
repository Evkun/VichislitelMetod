
import matplotlib.pyplot as plt
import numpy as np
def fun(x):
    return 2**x+5*x-3
e=0.0001
a=-0.5
b=1
x = np.arange(-2,2,0.01)  
y =  2**x+5*x-3
x2=np.arange(-2,2,0.01)  
y2=x*0 
plt.plot(x,y)   
plt.plot(x2,y2) 
   
plt.title("Line graph")   
plt.ylabel('Y axis')   
plt.xlabel('X axis')   
plt.show()   
c=0
iterr=0
while(abs(fun(a)*fun(b)) > e):
    iterr+=1
    if fun(a)*fun(b)>0:
        print("net resheniy")
        break
    else:
        c=(a+b)/2
        if fun(a)*fun(c)<0:
            b=c
        else:
            if fun(b)*fun(c)<0:
                a=c    
print("otvet",c)
print("chislo iteraciy",iterr)
def fx(x):
    return (2**x)*np.log(2)+5
x0=0.5
x1=x0-fun(x0)/fx(x0)
it=0
while(abs(x1-x0)>e):
    it+=1
    x0=x1
    x1=x0-fun(x0)/fx(x0)
print("Nyuton", x1)
print('it', it)
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

class MMT:
    def __init__(self, lr=1, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
    
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.v[key] = (self.momentum * self.v[key]) - self.lr * grads[key]
            params[key] += self.v[key]

# def f(x, y):
#     return (x**2 + y**2) + 5*np.sin(x + y**2)

# def df(x, y):
#     return 2*x + 5*np.cos(x + y**2), 2*y + 10*y*np.cos(x + y**2)

def f(x, y):
    return (x**2 + y**2) + 5*np.sin(x + 2*y**2)

def df(x, y):
    return 2*x + 5*np.cos(x + 2*y**2), 2*y + 20*y*np.cos(x + 2*y**2)      
    
init_pos = (5.0, 5.0)
params = {}
params['x'], params['y'] = init_pos[0], init_pos[1]
grads = {}
grads['x'], grads['y'] = 0, 0

norm_1 = 0
norm_2 = 0
dis = 0
dis_2 = 0

opt_x = -1.11
opt_y = 0

opt = MMT(lr=0.0213)

x_history = []
y_history = []
x_dot = []
y_dot = []
params['x'], params['y'] = init_pos[0], init_pos[1]

x_dot.append(init_pos[0])
y_dot.append(init_pos[1])

for i in range(50):
    x_history.append(params['x'])
    y_history.append(params['y'])
    
    grads['x'], grads['y'] = df(params['x'], params['y'])
    opt.update(params, grads)
    
    dis += ((params['x'] - opt_x) ** 2 + (params['y'] - opt_y) ** 2) ** (1/2)
    dis_2 += dis ** 2
    
    if i % 5 == 0:
        x_dot.append(params['x'])
        y_dot.append(params['y'])
        
norm_1 = dis
norm_2 = dis ** (1/2)

print("1norm = ", norm_1)
print("2norm = ", norm_2)

x_dot.append(x_history[-1])
y_dot.append(y_history[-1])

x = np.arange(-10, 10, 0.01)
y = np.arange(-10, 10, 0.01)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

mask = Z > 25 
Z[mask] = 0

plt.contour(X, Y, Z)
plt.plot(x_history, y_history, '-', color='red')  # 전체 최적화 경로를 선으로 이어줌
plt.plot(x_dot, y_dot, 'o', color = "black", markersize = 5)
plt.ylim(-5, 5.0)
plt.xlim(-5, 5)
plt.plot(-1.11, 0, 'x')
plt.title("MMT : f(x,y) = (x**2 + y**2) + 5sin(x + 2*y**2)\nlr = 0.0213      init_point(5, 5)")
plt.xlabel("x")
plt.ylabel("y")
print(f"x_history[-1]: {x_history[-1]:.4f}, y_history[-1]: {y_history[-1]:.4f}")
plt.show()


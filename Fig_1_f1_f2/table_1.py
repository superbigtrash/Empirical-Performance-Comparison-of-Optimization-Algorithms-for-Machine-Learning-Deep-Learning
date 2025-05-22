import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from optimizer import *

optimizers = {}
optimizers['SGD'] = {
    'optimizer': SGD(lr=0.07),
    'xhistory': [],
    'yhistory': [],
    'xdot' : [],
    'ydot' : []
}
optimizers['Momentum'] = {
    'optimizer': Momentum(lr=0.0213),
    'xhistory': [],
    'yhistory': [],
    'xdot' : [],
    'ydot' : []
}
optimizers['Nesterov'] = {
    'optimizer': Nesterov(lr=0.06),
    'xhistory': [],
    'yhistory': [],
    'xdot' : [],
    'ydot' : []
}
optimizers['AdaGrad'] = {
    'optimizer': AdaGrad(lr=2.8),
    'xhistory': [],
    'yhistory': [],
    'xdot' : [],
    'ydot' : []
}
optimizers['RMSprop'] = {
    'optimizer': RMSprop(lr=0.35),
    'xhistory': [],
    'yhistory': [],
    'xdot' : [],
    'ydot' : []
}
optimizers['Adam'] = {
    'optimizer': Adam(lr=2.175),
    'xhistory': [],
    'yhistory': [],
    'xdot' : [],
    'ydot' : []
}
optimizers['mSGD'] = {
    'optimizer': mSGD(),
    'xhistory': [],
    'yhistory': [],
    'xdot' : [],
    'ydot' : []
}

# def f(x, y):
#     return (x**2 + y**2) + 5*np.sin(x + y**2)

# def df(x, y):
#     return 2*x + 5*np.cos(x + y**2), 2*y + 10*y*np.cos(x + y**2)

def f(x, y):
    return (x**2 + y**2) + 5*np.sin(x + 2*y**2)

def df(x, y):
    return 2*x + 5*np.cos(x + 2*y**2), 2*y + 20*y*np.cos(x + 2*y**2)      

norm_1 = 0
dis = 0

opt_x, opt_y = -1.11, 0

for key in optimizers.keys():
    init_pos = (5.0, 5.0)
    params = {'x': init_pos[0], 'y': init_pos[1]}
    grads = {'x': 0, 'y': 0}
    dis = 0

    optimizers[key]['xdot'].append(init_pos[0])
    optimizers[key]['ydot'].append(init_pos[1])

    for i in range(50):
        optimizers[key]['xhistory'].append(params['x'])
        optimizers[key]['yhistory'].append(params['y'])

        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizers[key]['optimizer'].update(params, grads)

        dis += ((params['x'] - opt_x)**2 + (params['y'] - opt_y)**2)**0.5

        if i % 5 == 0:
            optimizers[key]['xdot'].append(params['x'])
            optimizers[key]['ydot'].append(params['y'])

    print(f"[{key}] 1-norm distance: {dis:.4f}")

    optimizers[key]['xdot'].append(optimizers[key]['xhistory'][-1])
    optimizers[key]['ydot'].append(optimizers[key]['yhistory'][-1])

x = np.arange(-10, 10, 0.01)
y = np.arange(-10, 10, 0.01)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

Z[Z > 25] = 0

for key in optimizers.keys():
    plt.contour(X, Y, Z)
    plt.plot(optimizers[key]['xhistory'], optimizers[key]['yhistory'], '-', color='red')
    plt.plot(optimizers[key]['xdot'], optimizers[key]['ydot'], 'o', color='black', markersize=5)
    plt.plot(opt_x, opt_y, 'x', color='blue')  
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.xlabel("x")
    plt.ylabel("y")
    if not optimizers['mSGD']:
        plt.title(f"{key} : f(x,y) = (x**2 + y**2) + 5sin(x + 2y**2)\nlr = {optimizers[key]['optimizer'].lr}, init=(5,5)")
    if optimizers['mSGD']:
        plt.title(f"{key} : f(x,y) = (x**2 + y**2) + 5sin(x + 2y**2)\n init=(5,5)")
        
    print(f"[{key}] last position: x = {optimizers[key]['xhistory'][-1]:.4f}, y = {optimizers[key]['yhistory'][-1]:.4f}")
    plt.show()


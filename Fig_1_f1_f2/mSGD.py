import numpy as np
import matplotlib.pyplot as plt

class mSGD():
    
    def __init__(self, k = 3, a = 0.3):
        self.k = k
        self.a = a
        
    def update(self, params, grads):
        
        used_lr = np.zeros(self.k)
        
        grads['x'], grads['y'] = df(params['x'], params['y'])
       
        for i in range(self.k):
            params_x = np.zeros(self.k)
            params_y = np.zeros(self.k)
            compare_min = np.zeros(self.k)
            self.rand_lr = np.random.uniform(0, self.a, size=self.k)
            
            for j in range(self.k):
                params_x[j] = params['x'] - self.rand_lr[j] * grads['x']
                params_y[j] = params['y'] - self.rand_lr[j] * grads['y']
                compare_min[j] = np.sqrt(params_x[j]**2 + params_y[j]**2)
                used_lr[j] = self.rand_lr[j]
                
            min_idx = np.argmin(compare_min)
            min_lr = used_lr[min_idx]

            for key in params.keys():
                params[key] -= min_lr * grads[key]

            return params['x'], params['y']  # 최적화된 위치 반환

# def f(x, y):
#     return (x**2 + y**2) + 5 * np.sin(x + y**2)

# def df(x, y):
#     return np.array([2 * x + 5 * np.cos(x + y**2), 2 * y + 10 * y * np.cos(x + y**2)])
        
def f(x, y):
    return (x**2 + y**2) + 5 * np.sin(x + 2*y**2)

def df(x, y):
    return np.array([2 * x + 5 * np.cos(x + 2*y**2), 2 * y + 20 * y * np.cos(x + 2*y**2)])

#################################################################################
init_pos = (5.0, 5.0) ##################################change###################
#################################################################################
params = {'x': init_pos[0], 'y': init_pos[1]}
grads = {'x': 0, 'y': 0}

norm_1 = 0
norm_2 = 0
dis = 0
dis_2 = 0

opt_x = -1.11
opt_y = 0

opt = mSGD()
x_history = []
y_history = []
x_dot = []
y_dot = []

x_dot.append(init_pos[0])
y_dot.append(init_pos[1])

x_history.append(init_pos[0])
y_history.append(init_pos[1])

# 최적화 과정 기록
for i in range(50):
    params['x'], params['y'] = opt.update(params, grads)
    x_history.append(params['x'])
    y_history.append(params['y'])
    
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
plt.plot(x_history, y_history, '-', color='red', linewidth = 2)  # 전체 최적화 경로를 선으로 이어줌
plt.plot(x_dot, y_dot, 'o', color = "black", markersize = 4)
plt.ylim(-5, 5)
plt.xlim(-5, 5)
plt.plot(-1.11, 0, 'x')
plt.title("mSGD : f(x,y) = (x**2 + y**2) + 5sin(x + 2*y**2)\nlr = random      init_point(5, 5)")
plt.xlabel("x")
plt.ylabel("y")
final_x, final_y = x_history[-1], y_history[-1]
print(f"final position: ({final_x:.4f}, {final_y:.4f})")
plt.show()


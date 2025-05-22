# coding: utf-8
import numpy as np

class SGD:

    """확률적 경사 하강법（Stochastic Gradient Descent）"""

    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key] 


class Momentum:

    """모멘텀 SGD"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():                                
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 
            params[key] += self.v[key]


class Nesterov:

    """Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)"""
    # NAG는 모멘텀에서 한 단계 발전한 방법이다. (http://newsight.tistory.com/224)
    
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


class AdaGrad:

    """AdaGrad"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSprop:

    """RMSprop"""

    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

class mSGD():
    
    def __init__(self, k = 3, a = 0.3):
        self.k = k
        self.a = a
        
    def update(self, params, grads):
        
        used_lr = np.zeros(self.k)
       
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

class KO:
    def __init__(self, params, lr=0.01, weight_decay=0, sigma_Q=0.01, sigma_R=2, gamma=5, momentum=0.9):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay
        self.sigma_Q = sigma_Q
        self.sigma_R = sigma_R
        self.gamma = gamma
        self.momentum = momentum
        self.state = {key: {'x': np.copy(value), 'P': np.eye(value.size) * 0.01} for key, value in params.items()}

    def easy_filter(self, y_t, x_t, P_t, sigma_Q, sigma_R, gamma):
        x_t = gamma * x_t
        P_t = gamma**2 * P_t + sigma_Q
        r_t = y_t - x_t
        K_t = P_t / (sigma_R + P_t)
        x_t = x_t + K_t * r_t
        P_t = (1 - K_t) * P_t
        return x_t, P_t

    def step(self):
        for key, param in self.params.items():
            grad = param.grad if hasattr(param, 'grad') else np.zeros_like(param)
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad += self.weight_decay * param

            state = self.state[key]
            x_t = state['x']
            P_t = state['P']

            x_t, P_t = self.easy_filter(grad, x_t, P_t, self.sigma_Q, self.sigma_R, self.gamma)

            # Update parameter
            param -= self.lr * x_t

            # Save updated state
            state['x'] = x_t
            state['P'] = P_t





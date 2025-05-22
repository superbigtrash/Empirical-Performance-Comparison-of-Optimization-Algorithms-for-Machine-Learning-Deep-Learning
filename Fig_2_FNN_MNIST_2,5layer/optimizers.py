# coding: utf-8
import numpy as np
import torch

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

class KO:
    def __init__(self, lr = 0.01, sigma_Q = 1, sigma_R = 2, gamma = 1, c = 1):
        self.lr = lr
        self.sigma_Q = sigma_Q
        self.sigma_R = sigma_R
        self.gamma = gamma
        self.c = c
        
    def update(self, params, grads):
        
        g_t = {}
        P_t = {}
        K_t = {}
        
        for key, values in params.items():
            g_t[key] = np.zeros_like(values)
            P_t[key] = np.zeros_like(values)
            K_t[key] = np.zeros_like(values)
        
        #y_t == grads
        
        for key, values in params.items():
            if params[key].ndim == 2:
                m, n = values.shape
                P_t[key] = np.random.randn(m, n) * 0.01
            elif params[key].ndim == 1:
                P_t[key] = np.zeros_like(values)
        
        #prediction step
        for key in g_t.keys():
            g_t[key] = self.gamma * g_t[key]
            P_t[key] = self.gamma**2 * P_t[key] + self.sigma_Q ** 2
        
            #update step
            K_t[key] = self.c * P_t[key] / (self.c ** 2 * P_t[key] + self.sigma_R ** 2)
            g_t[key] = g_t[key] + K_t[key] * (grads[key] - self.c * g_t[key])
            P_t[key] = (1 - self.c * P_t[key] * P_t[key])
            
            params[key] -= self.lr * g_t[key]
            
        
            
class mSGD:
    def __init__(self, point = 3, lr_range = 0.3, ith = 1):
        self.point = point
        self.ith = ith
        self.lr_range = lr_range
        self.min_lr = None
        
    def update(self, params, grads, x, autoencoder, iteration):
        self.params_box = {}
        self.lr_loss = np.zeros([self.point,2])
        self.lr_loss[:,0] = np.random.uniform(0, self.lr_range, self.point)
        
        if iteration == 0 or iteration % self.ith == 0:
            origin_params = params.copy()
            
            for i in range(self.point):
                
                _, x_hat = autoencoder.forward(x)
                
                for key in params.keys():
                    params[key] -= self.lr_loss[i,0] * grads[key]
                
                self.params_box[self.lr_loss[i,0]] = params  
                self.lr_loss[i,1] = autoencoder.loss(x, x_hat)
                
                params = origin_params.copy()
            
            min_index = np.argmin(self.lr_loss[:,1])           
            self.min_lr = self.lr_loss[min_index, 0]
            
            params = self.params_box[self.lr_loss[min_index, 0]]

        else:
            for key in params.keys():
                params[key] -= self.min_lr * grads[key]

        return autoencoder.loss(x, x_hat)
            


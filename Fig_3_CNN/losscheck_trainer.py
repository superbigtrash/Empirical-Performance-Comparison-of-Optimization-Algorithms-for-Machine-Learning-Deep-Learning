# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from optimizers import *
import time

class Trainer:
    """신경망 훈련을 대신 해주는 클래스
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 max_iter=2000, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test

        self.batch_size = mini_batch_size

        # optimzer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprop':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        
        self.train_size = x_train.shape[0]

        self.loss_norm_1 = 0
        
        self.max_iter = max_iter
        self.current_iter = 0
        
        self.loss_values = []
        self.optimizer_times = []

    def train_step(self):
        
        start_time = time.time()
        
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        
        loss = self.network.loss(x_batch, t_batch)
        self.loss_values.append(loss)
        
        end_time = time.time()
        self.optimizer_times.append(end_time - start_time)
        if self.current_iter > 950:
            self.loss_norm_1 += loss
        
        if self.current_iter % 100 == 0:
            print(f"Iteration {self.current_iter} - Train loss: {loss:.5f} - Time: {self.optimizer_times[-1]:.5f}s")
        
        self.current_iter += 1
        
        # If max iterations are reached, print the final loss
        if self.current_iter == self.max_iter:
            print(f"Iteration {self.current_iter} - Train loss: {loss:.5f} - Time: {self.optimizer_times[-1]:.5f}s")
        
    def train(self):
        print(self.optimizer, "starts!")
        for i in range(self.max_iter):
            self.train_step()
        print(self.optimizer, 'ends!')


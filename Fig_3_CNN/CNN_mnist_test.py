import sys, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mnist import load_mnist
from simple_convnet import SimpleConvNet
from losscheck_trainer import Trainer
from optimizers import *
from util import smooth_curve
from keras.datasets import cifar10
from keras.utils import to_categorical
import time
import pickle
from datetime import datetime

TF_ENABLE_ONEDNN_OPTS=0

###############################################################################
# mSGD_Trainer 따로 정의
class mSGD_Trainer:
    """신경망 훈련을 대신 해주는 클래스
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 max_iter=2000, mini_batch_size=100, m = 5, a = 0.3, optimizer='mSGD', verbose=True):
        
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test

        self.batch_size = mini_batch_size

        self.optimizer = optimizer
        
        self.m = m
        self.a = a
        
        self.train_size = x_train.shape[0]

        self.max_iter = max_iter
        self.current_iter = 0
        
        self.mSGD_params_container = {}
        self.loss_values = []
        self.optimizer_times = []
        self.selected_lr = 0
        self.loss_norm_1  = 0

    def train_step(self):
        
        start_time = time.time()  # Start timing for mSGD
        
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        if self.current_iter == 0 or self.current_iter % 1 == 0:
            loss_value=np.zeros([self.m,2])						# m-point mSGD
            loss_value[:,0] = np.random.uniform(0, self.a,self.m)
            # network.gradient(x_batch, t_batch)
            mSGD_grads = self.network.gradient(x_batch, t_batch)
    
            for j in range(self.m):
                origin_params = self.network.params.copy()
                for key in self.network.params.keys():
                    self.network.params[key] -= loss_value[j,0]*mSGD_grads[key]
               
                    self.mSGD_params_container[loss_value[j,0]] = self.network.params
                    # loss
                    loss_value[j,1] = self.network.loss(x_batch,t_batch)   
                    
                self.network.params = origin_params.copy()
                    
            # 가장 작은 loss의 lr 선택
            min_index=np.argmin(loss_value[:,1])
            self.selected_lr = loss_value[min_index, 0] 
            # 가장 loss가 작은 lr로 업데이트한 파라미터를 원래 모델에 다시 적용
            self.network.params = self.mSGD_params_container[loss_value[min_index,0]]
            
            if self.current_iter > 950:
                self.loss_norm_1 += loss_value[min_index, 1]
        else:
            mSGD_grads = self.network.gradient(x_batch, t_batch) 
            
            for key in self.network.params.keys():
                self.network.params[key] -= 0.1 * mSGD_grads[key]
           
        loss = self.network.loss(x_batch, t_batch)
        self.loss_values.append(loss)
        
        end_time = time.time()  # End timing for mSGD
        self.optimizer_times.append(end_time - start_time)

    
        if self.current_iter % 100 == 0:
            print(f"Iteration {self.current_iter} - Train loss: {loss:.5f} - Time: {self.optimizer_times[-1]:.5f}s")
        
        self.current_iter += 1
            
        # If max iterations are reached, print the final loss
        if self.current_iter == self.max_iter:
            print(f"Iteration {self.current_iter} - Train loss: {loss:.5f} - Time: {self.optimizer_times[-1]:.5f}s")
        
    def train(self):
        print("mSGD starts!")
        for i in range(self.max_iter):
            self.train_step()
        print("mSGD ends!")

#mSGD_Trainer 정의 끝부분 
################################################################################
class KO_Trainer:
    """신경망 훈련을 대신 해주는 클래스
    """
    def __init__(self, network, x_train, t_train, x_test, t_test, sigma2_Q = 1, sigma2_R = 2, gamma= 1, c = 1,
                 max_iter=2000, mini_batch_size=100, optimizer='KO', optimizer_param={'lr':1}, verbose=True):
        
        self.optimizer_param = optimizer_param
        self.sigma2_Q = sigma2_Q
        self.sigma2_R = sigma2_R
        self.gamma = gamma 
        self.c = c
        
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test

        self.batch_size = mini_batch_size

        self.optimizer = optimizer
        
        self.train_size = x_train.shape[0]

        self.loss_norm_1 = 0
        
        self.max_iter = max_iter
        self.current_iter = 0
        
        self.loss_values = []
        self.KO_g_t = {}
        for key, values in self.network.params.items():
                self.KO_g_t[key] = np.zeros_like(values)
         
        self.KO_P_t = {}
        #for key, values in network_KO.params.items():
        #    KO_P_t[key] = np.ones_like(values)*0.01

        for key, values in self.network.params.items():
            if self.network.params[key].ndim == 4:
                self.p, self.o, self.m, self.n = values.shape
                self.KO_P_t[key] = np.random.randn(self.m, self.n)*0.01
            if self.network.params[key].ndim == 3:
                self.o, self.m, self.n = values.shape
                self.KO_P_t[key] = np.random.randn(self.m, self.n)*0.01
            elif self.network.params[key].ndim == 2:
                self.m, self.n = values.shape
                self.KO_P_t[key] = np.random.randn(self.m,self.n)*0.01
            elif self.network.params[key].ndim == 1:    
                self.KO_P_t[key] = np.zeros_like(values)
            else:
                continue
          
        self.KO_K_t = {}
        for key, values in self.network.params.items():
            self.KO_K_t[key] = np.zeros_like(values)
            
        self.optimizer_times = []
        
    def train_step(self):  
        
        start_time = time.time()
        
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]
        
        y_t = self.network.gradient(x_batch, t_batch)   #1
                 
        for key in self.KO_g_t.keys():
            # prediction (propagation)
            self.KO_g_t[key] = self.gamma * self.KO_g_t[key]  
            self.KO_P_t[key] = self.gamma ** 2 * self.KO_P_t[key] + self.sigma2_Q    
        
            # update (correction)
    #        e_t[key] = y_t[key] - KO_c * KO_g_t[key]
            self.KO_K_t[key] = self.c * self.KO_P_t[key] / (self.c ** 2 * self.KO_P_t[key] + self.sigma2_R)   
            self.KO_g_t[key] = self.KO_g_t[key] + self.KO_K_t[key] * (y_t[key] - self.c * self.KO_g_t[key])                      
            self.KO_P_t[key] = (1 - self.c * self.KO_P_t[key]) * self.KO_P_t[key]                   
            
            self.network.params[key] -= self.optimizer_param['lr'] * self.KO_g_t[key]   # Update the parameters

        loss = self.network.loss(x_batch, t_batch)         # Compute loss
        self.loss_values.append(loss) 

        end_time = time.time()  # End timing for KO
        self.optimizer_times.append(end_time - start_time)

        if self.current_iter % 100 == 0:
            print(f"Iteration {self.current_iter} - Train loss: {loss:.5f} - Time: {self.optimizer_times[-1]:.5f}s")
        
        if self.current_iter > 950:
            self.loss_norm_1 += loss
        
        self.current_iter += 1
        
        # If max iterations are reached, print the final loss
        if self.current_iter == self.max_iter:
            print(f"Iteration {self.current_iter} - Train loss: {loss:.5f} - Time: {self.optimizer_times[-1]:.5f}s")
        
    def train(self):
        print(self.optimizer, "starts!")
        for i in range(self.max_iter):
            self.train_step()
        print(self.optimizer, 'ends!')
################################################################################
# 데이터 읽기
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

(x_train, t_train), (x_test, t_test) = cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

t_train = to_categorical(t_train, 10)
t_test = to_categorical(t_test, 10)

x_train = np.transpose(x_train, (0, 3, 1, 2))
x_test = np.transpose(x_test, (0, 3, 1, 2))

#input_dim

x = 3
y = 32
z = 32

hidden_sizes = 100
#mnist : x = 1, y = 28, z = 28
#cifar10 : x = 3, y = 32, z = 32 

# 시간이 오래 걸릴 경우 데이터를 줄인다.
#x_train, t_train = x_train[:5000], t_train[:5000]
#x_test, t_test = x_test[:1000], t_test[:1000]


network_1 = SimpleConvNet(input_dim=(x,y,z), 
                        conv_param = {'filter_num': 30, 'filter_size': 3, 'pad': 0, 'stride': 1},
                        hidden_size=hidden_sizes, output_size=10, weight_init_std=0.01)

network_2 = SimpleConvNet(input_dim=(x,y,z), 
                        conv_param = {'filter_num': 30, 'filter_size': 3, 'pad': 0, 'stride': 1},
                        hidden_size=hidden_sizes, output_size=10, weight_init_std=0.01)

network_3 = SimpleConvNet(input_dim=(x,y,z), 
                        conv_param = {'filter_num': 30, 'filter_size': 3, 'pad': 0, 'stride': 1},
                        hidden_size=hidden_sizes, output_size=10, weight_init_std=0.01)

network_4 = SimpleConvNet(input_dim=(x,y,z), 
                        conv_param = {'filter_num': 30, 'filter_size': 3, 'pad': 0, 'stride': 1},
                        hidden_size=hidden_sizes, output_size=10, weight_init_std=0.01)

network_5 = SimpleConvNet(input_dim=(x,y,z), 
                        conv_param = {'filter_num': 30, 'filter_size': 3, 'pad': 0, 'stride': 1},
                        hidden_size=hidden_sizes, output_size=10, weight_init_std=0.01)

network_6 = SimpleConvNet(input_dim=(x,y,z), 
                        conv_param = {'filter_num': 30, 'filter_size': 3, 'pad': 0, 'stride': 1},
                        hidden_size=hidden_sizes, output_size=10, weight_init_std=0.01)

network_mSGD = SimpleConvNet(input_dim=(x,y,z), 
                        conv_param = {'filter_num': 30, 'filter_size': 3, 'pad': 0, 'stride': 1},
                        hidden_size=hidden_sizes, output_size=10, weight_init_std=0.01)

network_KO = SimpleConvNet(input_dim=(x,y,z), 
                        conv_param = {'filter_num': 30, 'filter_size': 3, 'pad': 0, 'stride': 1},
                        hidden_size=hidden_sizes, output_size=10, weight_init_std=0.01)

##########################################################################################################

optimizers = {}
optimizers['SGD'] = 'SGD'
optimizers['Momentum'] = 'Momentum'
optimizers['Nesterov'] = 'Nesterov'
optimizers['AdaGrad'] = 'AdaGrad'
optimizers['RMSprop'] = 'RMSprop'
optimizers['Adam'] = 'Adam'

 
opts_lr = {}
opts_lr['SGD'] = 0.1
opts_lr['Momentum'] = 0.01
opts_lr['Nesterov'] = 0.01
opts_lr['AdaGrad'] = 0.01
opts_lr['RMSprop'] = 0.01
opts_lr['Adam'] = 0.001
opts_lr['KO'] = 1

loss_norm_1_values = {}
loss_norm_1_values['SGD'] = 0
loss_norm_1_values['Momentum'] = 0
loss_norm_1_values['Nesterov'] = 0
loss_norm_1_values['AdaGrad'] = 0
loss_norm_1_values['RMSprop'] = 0
loss_norm_1_values['Adam'] = 0

iterations = 1000
batch_size = 128

opts_loss_values = {}
opts_time = {}

point = 3
range_a = 0.1

###########################################################################

mSGD_start_time = time.time()

mSGD_trainer = mSGD_Trainer(network_mSGD, x_train, t_train, x_test, t_test,
                       max_iter=iterations, mini_batch_size=batch_size, m = point, a = range_a, optimizer='mSGD')
    
mSGD_trainer.train()

opts_loss_values['mSGD'] = mSGD_trainer.loss_values
opts_time['mSGD'] = mSGD_trainer.optimizer_times

mSGD_end_time = time.time()

mSGD_total_time = mSGD_end_time - mSGD_start_time
print("total time : ", mSGD_total_time)
print("avg time or total/iteration : ", mSGD_total_time/iterations)

##########################################################################

for opt_key, lr_key in zip(optimizers.keys(), opts_lr.keys()):      
    opt_start_time = time.time()
    network = eval(f"network_{list(optimizers.keys()).index(opt_key) + 1}")                
    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                        max_iter=iterations, mini_batch_size=batch_size,
                        optimizer=optimizers[opt_key], optimizer_param={'lr': opts_lr[lr_key]})
    trainer.train()
    
    opt_end_time = time.time()
    
    opt_total_time = opt_end_time - opt_start_time
    
    print("total time : ", opt_total_time)
    print("avg time or total/iteration : ", opt_total_time/iterations)
    
    opts_loss_values[opt_key] = trainer.loss_values
    opts_time[opt_key] = trainer.optimizer_times
    loss_norm_1_values[opt_key] = trainer.loss_norm_1

##########################################################################

KO_start_time = time.time()

KO_trainer = KO_Trainer(network_KO, x_train, t_train, x_test, t_test, 
                        sigma2_Q = 1, sigma2_R = 2, gamma= 1, c = 1,
                        max_iter = iterations, mini_batch_size = batch_size, optimizer = 'KO', 
                        optimizer_param = {'lr':opts_lr['KO']})
    
KO_trainer.train()

KO_end_time = time.time()

KO_total_time = KO_end_time - KO_start_time
print("total time : ", KO_total_time)
print("avg time or total/iteration : ", KO_total_time/iterations)

opts_loss_values['KO'] = KO_trainer.loss_values
opts_time['KO'] = KO_trainer.optimizer_times


# 그래프 그리기
markers = {"SGD": "x", "Momentum": ">", "Nesterov":"^", "AdaGrad": "v", "RMSprop": "s", "Adam": "*", "mSGD": "o", "KO" : "D"}

optimizers['KO'] = 'KO'
optimizers['mSGD'] = 'mSGD'

for key in optimizers.keys():
    avg_time = np.mean(opts_time[key])
    total_time = np.sum(opts_time[key])
    print(f"{key} - Average time per iteration: {avg_time:.5f}s, Total time: {total_time:.2f}s")

x = np.arange(iterations)

for key in optimizers.keys():    
    plt.plot(x, smooth_curve(opts_loss_values[key]), marker=markers[key], markevery=200,  label=key)

plt.title("CNN_CIFAR10")
plt.ylim(1.3, 2.4)
plt.xlabel("iterations")
plt.ylabel("loss values")
plt.legend()
plt.show()

for key in optimizers.keys():    
    plt.plot(x, smooth_curve(opts_loss_values[key]), marker=markers[key], markevery=200,  label=key)

plt.title("CNN_CIFAR10")
plt.ylim(1.3, 1.8)
plt.xlabel("iterations")
plt.ylabel("loss values")
plt.legend()
plt.show()

def moving_average(data, window_size=3):
    """간단한 이동 평균 계산 함수"""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

for key in optimizers.keys():
    # 이동 평균 적용
    y_moving_avg = moving_average(smooth_curve(opts_loss_values[key]), window_size=50)
    x_moving_avg = x[:len(y_moving_avg)]  # x 데이터 크기를 y와 맞춤

    plt.plot(x_moving_avg, y_moving_avg, marker=markers[key], markevery=100, label=key)

plt.title("CNN_CIFAR10 (50-point Moving Average)")
plt.xlabel("iterations")
plt.ylabel("loss values")
plt.ylim(1.3, 2.4)
plt.legend()
plt.show()

for key in optimizers.keys():
    # 이동 평균 적용
    y_moving_avg = moving_average(smooth_curve(opts_loss_values[key]), window_size=50)
    x_moving_avg = x[:len(y_moving_avg)]  # x 데이터 크기를 y와 맞춤

    plt.plot(x_moving_avg, y_moving_avg, marker=markers[key], markevery=100, label=key)

plt.title("CNN_CIFAR10 (50-Point Moving Average)")
plt.xlabel("iterations")
plt.ylabel("loss values")
plt.ylim(1.3, 1.8)
plt.legend()
plt.show()


for opt_key, value in loss_norm_1_values.items():
    print(f"{opt_key}: {value}")

print("KO L1 Norm 값:", KO_trainer.loss_norm_1)    
print("mSGD L1 Norm 값:", mSGD_trainer.loss_norm_1)

# loss_norm_1_values 딕셔너리 기준으로 오름차순 정렬
sorted_loss_norm = sorted(loss_norm_1_values.items(), key=lambda x: x[1])

# KO_trainer와 mSGD_trainer 값도 같이 리스트에 넣고 정렬
extra_norms = {
    "KO_trainer": KO_trainer.loss_norm_1,
    "mSGD_trainer": mSGD_trainer.loss_norm_1
}

all_norms = list(loss_norm_1_values.items()) + list(extra_norms.items())

# 전체 오름차순 정렬
sorted_all_norms = sorted(all_norms, key=lambda x: x[1])

print("\n모든 norm 값 오름차순 정렬:")
for name, val in sorted_all_norms:
    print(f"{name}: {val}")
    
# 결과 저장할 딕셔너리 생성
training_results = {
    "loss_values": opts_loss_values,
    "time_values": opts_time,
    "loss_norm_values": loss_norm_1_values,
    "KO_loss_norm": KO_trainer.loss_norm_1,
    "mSGD_loss_norm": mSGD_trainer.loss_norm_1
}

# 현재 시간으로 파일명 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"training_results_{timestamp}.pkl"

# Pickle 파일로 저장
with open(filename, "wb") as f:
    pickle.dump(training_results, f)
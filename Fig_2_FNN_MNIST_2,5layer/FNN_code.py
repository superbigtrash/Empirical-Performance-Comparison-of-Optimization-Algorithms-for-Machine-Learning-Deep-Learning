import os
import sys
sys.path.append(os.pardir)  	
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
from datetime import datetime

from mnist import load_mnist
from util import smooth_curve
from multi_layer_net import MultiLayerNet
from optimizers import *

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
train_size = x_train.shape[0]
batch_size = 128
max_iterations = 1000                 
p = 3                                               
a = 0.3

optimizers = {}
optimizers['SGD'] = SGD(lr = 0.1)
optimizers['Momentum'] = Momentum(lr = 0.1)
optimizers['Nesterov'] = Nesterov(lr = 0.1)
optimizers['AdaGrad'] = AdaGrad(lr = 0.1)
optimizers['RMSprop'] = RMSprop(lr = 0.01)
optimizers['Adam'] = Adam(lr = 0.01)

networks = {}
train_loss = {}
opts_time = {key: 0.0 for key in optimizers.keys()}
opts_time['KO'] = 0.0
opts_time['mSGD'] = 0.0

loss_sum_container = {}
loss_sum_container['SGD'] = 0
loss_sum_container['Momentum'] = 0
loss_sum_container['Nesterov'] = 0
loss_sum_container['AdaGrad'] = 0
loss_sum_container['RMSprop'] = 0
loss_sum_container['Adam'] = 0

mSGD_norm_1 = 0
KO_norm_1 = 0

layer = 1

for key in optimizers.keys():
    networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100] * layer, output_size=10)
    train_loss[key] = []    

network_mSGD = MultiLayerNet(input_size=784, hidden_size_list=[100] * layer, output_size=10)
train_loss['mSGD'] = []
mSGD_params_container = {}

network_KO = MultiLayerNet(input_size=784, hidden_size_list=[100] * layer, output_size=10)
train_loss['KO'] = []
KO_lr = 1
KO_sigma2_Q = 1
KO_sigma2_R = 2
KO_gamma = 1
KO_c = 1

KO_g_t = {}
for key, values in network_KO.params.items():
        KO_g_t[key] = np.zeros_like(values)
 
KO_P_t = {}

for key, values in network_KO.params.items():
    if network_KO.params[key].ndim == 2:
        m, n = values.shape
        KO_P_t[key] = np.random.randn(m,n)*0.01
    elif network_KO.params[key].ndim == 1:    
        KO_P_t[key] = np.zeros_like(values)
    else:
        continue
  
KO_K_t = {}
for key, values in network_KO.params.items():
    KO_K_t[key] = np.zeros_like(values)

for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    for key in optimizers.keys():                     
        start_time = time.time() 
        
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads) 
        loss = networks[key].loss(x_batch, t_batch)
        
        if i > 950: 
            loss_sum_container[key] += loss
        
        end_time = time.time()  
        opts_time[key] += (end_time - start_time) 
        
        train_loss[key].append(loss)

    mSGD_start_time = time.time()
    
    loss_value=np.zeros([p,2])				
    loss_value[:,0] = np.random.uniform(0, a, p) # modified part 1  distribution
    mSGD_grads = network_mSGD.gradient(x_batch, t_batch)
    origin_params = network_mSGD.params.copy()
    for j in range(p):
         for key in network_mSGD.params.keys():
             network_mSGD.params[key] -= loss_value[j,0]*mSGD_grads[key]
   
         mSGD_params_container[loss_value[j,0]] = network_mSGD.params
         loss_value[j,1] = network_mSGD.loss(x_batch,t_batch)       
         network_mSGD.params = origin_params.copy()
             
    min_index=np.argmin(loss_value[:,1])
    selected_lr = loss_value[min_index, 0] 
    train_loss['mSGD'].append(loss_value[min_index,1])
    network_mSGD.params = mSGD_params_container[loss_value[min_index,0]]
    
    mSGD_end_time = time.time()
    opts_time['mSGD'] += (mSGD_end_time - mSGD_start_time)

    if i > 950: 
        mSGD_norm_1 += loss_value[min_index, 1]

    KO_start_time = time.time()
    y_t = network_KO.gradient(x_batch, t_batch)  
     
    for key in KO_g_t.keys():
        KO_g_t[key] = KO_gamma * KO_g_t[key]  
        KO_P_t[key] = KO_gamma ** 2 * KO_P_t[key] + KO_sigma2_Q    
    
        KO_K_t[key] = KO_c * KO_P_t[key] / (KO_c ** 2 * KO_P_t[key] + KO_sigma2_R)   
        KO_g_t[key] = KO_g_t[key] + KO_K_t[key] * (y_t[key] - KO_c * KO_g_t[key])                      
        KO_P_t[key] = (1 - KO_c * KO_P_t[key]) * KO_P_t[key]                   
        
        network_KO.params[key] -= KO_lr * KO_g_t[key]   

    KO_loss = network_KO.loss(x_batch, t_batch) 
    KO_end_time = time.time()
    opts_time['KO'] += (KO_end_time - KO_start_time)
    
    train_loss['KO'].append(KO_loss) 

    if i > 950: 
        KO_norm_1 += KO_loss

    if i % 100 == 0 or i == max_iterations - 1:
        print( "===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))
                      
        # KO loss 추가
        loss = network_KO.loss(x_batch, t_batch)
        print(f'KO : {loss}')
        
        # mSGD loss 추가
        loss = network_mSGD.loss(x_batch, t_batch)
        print(f'mSGD : {loss}')
        
        print('='*50)
        for key, total_time in opts_time.items():
            print(f"{key}: {total_time:.5f} seconds")
      
optimizers['KO'] = 'KO'        
optimizers['mSGD'] = 'mSGD'

markers = {"SGD": "x", "Momentum": ">", "Nesterov":"^", "AdaGrad": "v", "RMSprop": "s", "Adam": "*", "KO" : "D", "mSGD": "o"}      

x = np.arange(max_iterations)

for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)

plt.title("FNN_MINST")       
plt.xlabel("iterations")
plt.ylabel("loss values")
plt.ylim(0, 1)
plt.legend()
plt.show()

for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)

plt.title("FNN_MINST")       
plt.xlabel("iterations")
plt.ylabel("loss values")
plt.ylim(0, 0.25)
plt.legend()
plt.show()


def moving_average(data, window_size=3):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

for key in optimizers.keys():
    y_moving_avg = moving_average(smooth_curve(train_loss[key]), window_size=50)
    x_moving_avg = x[:len(y_moving_avg)] 
    plt.plot(x_moving_avg, y_moving_avg, marker=markers[key], markevery=100, label=key)

plt.title("FNN_MINST (50 iteration moving average)")
plt.xlabel("iterations")
plt.ylabel("loss values")
plt.ylim(0, 1)
plt.legend()
plt.show()

for key in optimizers.keys():
    y_moving_avg = moving_average(smooth_curve(train_loss[key]), window_size=50)
    x_moving_avg = x[:len(y_moving_avg)]  

    plt.plot(x_moving_avg, y_moving_avg, marker=markers[key], markevery=100, label=key)

plt.title("FNN_MINST (50 iteration moving average)")
plt.xlabel("iterations")
plt.ylabel("loss values")
plt.ylim(0, 0.25)
plt.legend()
plt.show()

print(loss_sum_container)
print(mSGD_norm_1)
print(KO_norm_1)

#  현재 시간 정보 가져오기
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")

#  저장할 파일명 (타임스탬프 포함)
filename = f'optimizer_loss_values_{timestamp}.pkl'

#  저장할 데이터 정리
loss_data = {
    'train_loss': train_loss,
    'opts_time' : opts_time,
    'norm_1' : loss_sum_container,
    'norm_2' : loss_sum_container2
}

#  pickle 파일로 저장
with open(filename, 'wb') as f:
    pickle.dump(loss_data, f)
    
# Norm 값을 담는 딕셔너리 생성
norms = {**loss_sum_container, 'mSGD': mSGD_norm_1, 'KO': KO_norm_1}

# Norm 값의 오름차순 정렬
sorted_norms = sorted(norms.items(), key=lambda x: x[1])

# 결과 출력
print("\n=== Optimizer Loss Norm (오름차순) ===")
for optimizer, norm in sorted_norms:
    print(f"{optimizer}: {norm}")
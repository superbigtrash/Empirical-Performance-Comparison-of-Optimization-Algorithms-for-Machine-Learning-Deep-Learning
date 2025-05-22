import os
import sys
sys.path.append(os.pardir)  				# 부모 디렉터리의 파일을 가져올 수 있도록 설정
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
from datetime import datetime

from mnist import load_mnist
from util import smooth_curve
from autoencoder_structure import *
from optimizers import *


(x_train, _), (x_test, t_test) = load_mnist(normalize=True)

numbers = []
where = []
for i in range(10):  # from 0 to 9 sample add 
    for j in range(len(t_test)): 
        if i == t_test[j]:  
            where.append(j)
            numbers.append(i)
            break

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 1000                 
p = 3                                                   # m-point mSGD
a = 0.1

####################################################################
optimizers = {}
optimizers['SGD'] = SGD(lr = 0.1)
optimizers['Momentum'] = Momentum(lr = 0.01)
optimizers['Nesterov'] = Nesterov(lr = 0.01)
optimizers['AdaGrad'] = AdaGrad(lr = 0.01)
optimizers['RMSprop'] = RMSprop(lr = 0.001)
optimizers['Adam'] = Adam(lr = 0.001)

AutoEncoders = {}
train_loss = {}
loss_dict = {}

loss_sum_container = {}
loss_sum_container['SGD'] = 0
loss_sum_container['Momentum'] = 0
loss_sum_container['Nesterov'] = 0
loss_sum_container['AdaGrad'] = 0
loss_sum_container['RMSprop'] = 0
loss_sum_container['Adam'] = 0

mSGD_norm_1 = 0
KO_norm_1 = 0

opts_time = {key: 0.0 for key in optimizers.keys()}
opts_time['KO'] = 0.0
opts_time['mSGD'] = 0.0

for key in optimizers.keys():
    AutoEncoders[key] = AutoEncoder(input_size=784, hidden_size_list=[256, 256], latent_size=128)
    train_loss[key] = []    

########################################################################################
# mSGD autoencoder
autoencoder_mSGD = AutoEncoder(input_size=784, hidden_size_list=[256, 256], latent_size=128)
train_loss['mSGD'] = []
mSGD_params_container = {}
##################################################################################

########################################################################################
# KO autoencoder 
autoencoder_KO = AutoEncoder(input_size=784, hidden_size_list=[256, 256], latent_size=128)
train_loss['KO'] = []
KO_lr = 0.1
KO_sigma2_Q = 1
KO_sigma2_R = 2
KO_gamma = 1
KO_c = 1

KO_g_t = {}
for key, values in autoencoder_KO.params.items():
        KO_g_t[key] = np.zeros_like(values)
 
KO_P_t = {}

for key, values in autoencoder_KO.params.items():
    if autoencoder_KO.params[key].ndim == 2:
        m, n = values.shape
        KO_P_t[key] = np.random.randn(m,n)*0.01
    elif autoencoder_KO.params[key].ndim == 1:    
        KO_P_t[key] = np.zeros_like(values)
    else:
        continue
  
KO_K_t = {}
for key, values in autoencoder_KO.params.items():
    KO_K_t[key] = np.zeros_like(values)

 
########################################################################################
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask] 
    
    for key in optimizers.keys():                           # key: SGD, AdaGrad, Adam ...
        start_time = time.time()
        
        encoded, decoded = AutoEncoders[key].forward(x_batch)   
        grads = AutoEncoders[key].gradient(x_batch)
        optimizers[key].update(AutoEncoders[key].params, grads) 
        # Calculate loss
        loss_dict[key] = AutoEncoders[key].loss(x_batch, decoded)
        
        end_time = time.time() 
            
        if i > 950: 
            loss_sum_container[key] += loss_dict[key]
    
        opts_time[key] += (end_time - start_time)
        
        train_loss[key].append(loss_dict[key])
        
################################################################################ 
    mSGD_start_time = time.time()

    if i == 0 or i % 1 == 0:
        loss_value=np.zeros([p,2])						# m-point mSGD
        loss_value[:,0] = np.random.uniform(0, a, p)
        mSGD_grads = autoencoder_mSGD.gradient(x_batch)
        params_box = {}
        for j in range(p):
            # gradient.update()
            origin_params = autoencoder_mSGD.params.copy()
            
            _, decoded = autoencoder_mSGD.forward(x_batch)
            for key in autoencoder_mSGD.params.keys():
                 autoencoder_mSGD.params[key] -= loss_value[j,0]*mSGD_grads[key]
   
            mSGD_params_container[loss_value[j,0]] = autoencoder_mSGD.params
                # loss
            loss_value[j,1] = autoencoder_mSGD.loss(x_batch, decoded)   
             
            autoencoder_mSGD.params = origin_params.copy()
    
        min_index=np.argmin(loss_value[:,1])
        selected_lr = loss_value[min_index, 0] 
        mSGD_loss = loss_value[min_index,1]
        
        if i > 950: 
            mSGD_norm_1 += loss_value[min_index, 1]
        
        train_loss['mSGD'].append(mSGD_loss)
        autoencoder_mSGD.params = mSGD_params_container[loss_value[min_index,0]]

    else:
        _, decoded = autoencoder_mSGD.forward(x_batch)
        mSGD_grads = autoencoder_mSGD.gradient(x_batch)
        for key in autoencoder_mSGD.params.keys():
            autoencoder_mSGD.params[key] -= 0.1 * mSGD_grads[key]
        mSGD_loss = autoencoder_mSGD.loss(x_batch, decoded)
        train_loss['mSGD'].append(mSGD_loss)

    mSGD_end_time = time.time()
    opts_time['mSGD'] += (mSGD_end_time - mSGD_start_time)
    
########################################################################################   
    KO_start_time = time.time()

    _, decoded = autoencoder_KO.forward(x_batch)
    y_t = autoencoder_KO.gradient(x_batch)   #1
       
    for key in KO_g_t.keys():
        # prediction (propagation)
        KO_g_t[key] = KO_gamma * KO_g_t[key]  
        KO_P_t[key] = KO_gamma ** 2 * KO_P_t[key] + KO_sigma2_Q    
    
        # update (correction)
        KO_K_t[key] = KO_c * KO_P_t[key] / (KO_c ** 2 * KO_P_t[key] + KO_sigma2_R)   
        KO_g_t[key] = KO_g_t[key] + KO_K_t[key] * (y_t[key] - KO_c * KO_g_t[key])                      
        KO_P_t[key] = (1 - KO_c * KO_P_t[key]) * KO_P_t[key]                   
        
        autoencoder_KO.params[key] -= KO_lr * KO_g_t[key]   # Update the parameters

    KO_loss = autoencoder_KO.loss(x_batch, decoded)         # Compute loss
    
    if i > 950: 
        KO_norm_1 += KO_loss
    
    KO_end_time = time.time()
    opts_time['KO'] += (KO_end_time - KO_start_time)

    train_loss['KO'].append(KO_loss) 
########################################################################################

    if i % 100 == 0 or i == max_iterations - 1:
        print( "===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            losses = loss_dict[key]
            print(key + ":" + str(losses))
                      
        loss = KO_loss
        print(f'KO : {loss}')
        
        loss = mSGD_loss
        print(f'mSGD : {loss}')
        
        print('='*50)
        for key, total_time in opts_time.items():
            print(f"{key}: {total_time:.5f} seconds")

#####################################################################################
optimizers['KO'] = 'KO'                          
optimizers['mSGD'] = 'mSGD'
#####################################################################################

markers = {"SGD": "x", "Momentum": ">", "Nesterov":"^", "AdaGrad": "v", "RMSprop": "s", "Adam": "*", "KO" : "D", "mSGD": "o"}      

x = np.arange(max_iterations)

for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)

plt.title("AutoEncoder_MINST")         
plt.xlabel("iterations")
plt.ylabel("loss values")
plt.ylim(0.15, 0.3)
plt.legend()
plt.show()

x = np.arange(max_iterations)

for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)

plt.title("AutoEncoder_MINST")       
plt.xlabel("iterations")
plt.ylabel("loss values")
plt.ylim(0.15, 0.7)
plt.legend()
plt.show()

# Select a few test images (e.g., first 3 images from x_test)
num_images = 10
test_images = np.array([x_test[idx] for idx in where])

visual_optimizers = ['SGD', 'Momentum', 'Nesterov', 'AdaGrad', 'RMSprop', 'Adam']

fig, axes = plt.subplots(num_images, len(visual_optimizers) + 3, figsize=(12, 12))

# Display the original images in the first column
for i in range(num_images):
    axes[i, 0].imshow(test_images[i].reshape(28, 28), cmap="gray")
    axes[i, 0].set_title("Original")
    axes[i, 0].axis('off')  # Turn off axes

# display the reconstructed images for each selected optimizer
for idx, key in enumerate(visual_optimizers):
    for i in range(num_images):
        encoded, decoded = AutoEncoders[key].forward(test_images[i:i+1])
        axes[i, idx + 1].imshow(decoded.reshape(28, 28), cmap="gray")
        axes[i, idx + 1].set_title(f"{key}")
        axes[i, idx + 1].axis('off')  # Turn off axes

# KO results
encoded, decoded = autoencoder_KO.forward(test_images)
for i in range(num_images):
    axes[i, len(visual_optimizers) + 1].imshow(decoded[i].reshape(28, 28), cmap="gray")
    axes[i, len(visual_optimizers) + 1].set_title("KO")
    axes[i, len(visual_optimizers) + 1].axis('off')

# mSGD results
encoded, decoded = autoencoder_mSGD.forward(test_images)
for i in range(num_images):
    axes[i, len(visual_optimizers) + 2].imshow(decoded[i].reshape(28, 28), cmap="gray")
    axes[i, len(visual_optimizers) + 2].set_title("mSGD")
    axes[i, len(visual_optimizers) + 2].axis('off')

plt.tight_layout()
plt.show()

print(loss_sum_container)
print(mSGD_norm_1)
print(KO_norm_1)

#  current time info
now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")

#  file name 
filename = f'optimizer_loss_values_{timestamp}.pkl'

#  data for saving
loss_data = {
    'train_loss': train_loss,
    'opts_time' : opts_time,
    'norm_1' : loss_sum_container 
}

#  save pickle file 
with open(filename, 'wb') as f:
    pickle.dump(loss_data, f)
    
# dict for values
norms = {**loss_sum_container, 'mSGD': mSGD_norm_1, 'KO': KO_norm_1}

# sorting Norm value 
sorted_norms = sorted(norms.items(), key=lambda x: x[1])

# print out results
print("\n=== Optimizer Loss Norm ===")
for optimizer, norm in sorted_norms:
    print(f"{optimizer}: {norm}")

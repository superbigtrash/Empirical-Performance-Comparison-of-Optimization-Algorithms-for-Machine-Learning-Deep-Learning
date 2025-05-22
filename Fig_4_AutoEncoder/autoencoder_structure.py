import os
import sys
sys.path.append(os.pardir)  				# 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from layers import *
from collections import OrderedDict

class AutoEncoder:
    def __init__(self, input_size, hidden_size_list,latent_size):
        self.input_size = input_size
        self.hidden_size_list = hidden_size_list
        self.latent_size = latent_size
        self.weight_init_std = 0.01
        
        self.params = {}
        self.params['W1'] = self.weight_init_std * np.random.randn(input_size, hidden_size_list[0])
        self.params['b1'] = np.zeros(hidden_size_list[0])
        self.params['W2'] = self.weight_init_std * np.random.randn(hidden_size_list[0], hidden_size_list[1])
        self.params['b2'] = np.zeros(hidden_size_list[1])
        self.params['W3'] = self.weight_init_std * np.random.randn(hidden_size_list[1], latent_size)
        self.params['b3'] = np.zeros(latent_size)
        self.params['W4'] = self.weight_init_std * np.random.randn(latent_size, hidden_size_list[1])
        self.params['b4'] = np.zeros(hidden_size_list[1])
        self.params['W5'] = self.weight_init_std * np.random.randn(hidden_size_list[1], hidden_size_list[0])
        self.params['b5'] = np.zeros(hidden_size_list[0])
        self.params['W6'] = self.weight_init_std * np.random.randn(hidden_size_list[0], input_size)
        self.params['b6'] = np.zeros(input_size)
        
        self.encoder = OrderedDict()
        self.encoder['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.encoder['Relu1'] = Relu()
        self.encoder['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.encoder['Relu2'] = Relu()
        self.encoder['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        
        self.decoder = OrderedDict()
        self.decoder['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.decoder['Relu4'] = Relu()
        self.decoder['Affine5'] = Affine(self.params['W5'], self.params['b5'])
        self.decoder['Relu5'] = Relu()
        self.decoder['Affine6'] = Affine(self.params['W6'], self.params['b6'])
        self.decoder['Sigmoid'] = Sigmoid()

    def forward(self, x):
        for layer in self.encoder.values():
            x = layer.forward(x)
        encoding = x
        for layer in self.decoder.values():
            x = layer.forward(x)
        decoding = x
        return encoding, decoding
    
    def backward(self, dout):
        for layer in reversed(self.decoder.values()):
            dout = layer.backward(dout)
        for layer in reversed(self.encoder.values()):
            dout = layer.backward(dout)
        return dout
        
    def loss(self, x, x_hat):

        num_of_pixels = 28 * 28
        delta = 1e-7  

        sample_loss = -np.sum(x * np.log(x_hat + delta) + (1 - x) * np.log(1 - x_hat + delta), axis=1) / num_of_pixels

        return np.mean(sample_loss)

    def gradient(self, x):

        encoding, decoding = self.forward(x)

        dout = (decoding - x) / x.shape[0]
        self.backward(dout)

        grads = {}
    
        # Encoder gradients
        for idx in range(1, 4):
            grads['W' + str(idx)] = self.encoder[f'Affine{idx}'].dW
            grads['b' + str(idx)] = self.encoder[f'Affine{idx}'].db
    
        # Decoder gradients
        for idx in range(4, 7):
            grads['W' + str(idx)] = self.decoder[f'Affine{idx}'].dW
            grads['b' + str(idx)] = self.decoder[f'Affine{idx}'].db
            
        return grads

        
        
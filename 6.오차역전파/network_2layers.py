import numpy as np
from functions import *

def numerical_gradient_1d(f, x):
    h = 10**(-4)
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp = x[idx]
        x[idx] = float(tmp) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp # 값 복원
        
    return grad

def numerical_gradient_2d(f, X):
    if X.ndim == 1:
        return numerical_gradient(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient_1d(f, x)
        
        return grad


class Net_Cls:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size+1, hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size+1, output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        
        b = [1] if x.ndim==1 else np.ones((x.shape[0], 1))
        
        x = np.append(b, x, axis=x.ndim-1)
        a1 = np.dot(x, W1)
        z1 = np.append(b, sigmoid(a1), axis=x.ndim-1)
        a2 = np.dot(z1, W2)
        y = softmax(a2)
        
        return y
        
    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)        
        return cross_entropy_error(y, t)
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        
        grads = {}
        grads['W1'] = numerical_gradient_2d(loss_W, self.params['W1'])
        grads['W2'] = numerical_gradient_2d(loss_W, self.params['W2'])
        
        return grads  
    
    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        
        b = [1] if x.ndim==1 else np.ones((x.shape[0], 1))
        
        grads = {}        
        
        # forward
        x = np.append(b, x, axis=x.ndim-1)
        a1 = np.dot(x, W1)
        z1 = np.append(b, sigmoid(a1), axis=x.ndim-1)
        a2 = np.dot(z1, W2)
        y = softmax(a2)
        
        # backward
        dy = (y - t) / x.shape[0]
        grads['W2'] = np.dot(z1.T, dy)
        
        c = np.dot(dy, W2.T)
        c = tmp[1:] if x.ndim==1 else c[:, 1:]
        dz1 = sigmoid_grad(a1) * c
        grads['W1'] = np.dot(x.T, dz1)        

        return grads
    


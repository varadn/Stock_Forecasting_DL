import numpy as np

import numpy as np


class Softmax:
    def __init__(self):
        self.type = 'Softmax'
        self.eps = 1e-15

    def forward(self, Z):
        self.Z = Z

        t = np.exp(Z - np.max(Z, axis=0))
        self.A =  t / np.sum(t, axis=0, keepdims=True)

        return self.A


class Tanh:
    def __init__(self):
        self.type = 'Tanh'

    def forward(self, Z):
        self.A = np.tanh(Z)

        return self.A

    def backward(self, dA):
        dZ = dA * (1 - np.power(self.A, 2))

        return dZ


# Loss Function
class CrossEntropyLoss:
    def __init__(self):
        self.type = 'CELoss'
        self.eps = 1e-15
        self.softmax = Softmax()
    
    def forward(self, Y_hat, Y):
        self.Y = Y
        self.Y_hat = Y_hat

        _loss = - Y * np.log(self.Y_hat)
        loss = np.sum(_loss, axis=0).mean()

        return np.squeeze(loss) 

    def backward(self):
        grad = self.Y_hat - self.Y
        
        return grad


# Stochastic Gradient Descent
class SGD:
    def __init__(self, lr=0.0075, beta=0.9):
        self.beta = beta
        self.lr = lr

    def optim(self, weights, gradients, velocities=None):
        if velocities is None: velocities = [0 for weight in weights]

        velocities = self._update_velocities(
            gradients, self.beta, velocities
        )
        new_weights = []

        for weight, velocity in zip(weights, velocities):
            weight -= self.lr * velocity
            new_weights.append(weight)

        return new_weights, velocities

    def _update_velocities(self, gradients, beta, velocities):
        new_velocities = []

        for gradient, velocity in zip(gradients, velocities):

            new_velocity = beta * velocity + (1 - beta) * gradient
            new_velocities.append(new_velocity)

        return new_velocities


def one_hot_encoding(input, size):
    output = []

    for index, num in enumerate(input):
        one_hot = np.zeros((size,1))

        if (num != None):
            one_hot[num] = 1
    
        output.append(one_hot)

    return output
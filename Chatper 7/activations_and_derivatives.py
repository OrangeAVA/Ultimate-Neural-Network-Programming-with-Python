# sigmoid function Python Code
import numpy as np


def sigmoid(z):
  return 1.0 / (1 + np.exp(-z))


# Derivative of sigmoid function
def sigmoid_prime(z):
  return sigmoid(z) * (1-sigmoid(z))

# Tanh activation function Python Code
import numpy as np
def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


# Derivative of Tanh Activation Function
def tanh_prime(z):
    return 1 - np.power(tanh(z), 2)


# ReLU function Python Code
import numpy as np


# ReLU activation function
def relu(z):
  return max(0, z)


# Derivative of ReLU Activation Function
def relu_prime(z):
  return 1 if z > 0 else 0


# Leaky ReLU activation function
import numpy as np


def leaky_relu(z):
  alpha = 0.1
  return z if z > 0 else alpha*z


# Derivative of ReLU Activation Function
def leaky_relu_prime(z):
  alpha=0.1
  return 1 if z > 0 else alpha



import numpy as np


def rbf_gaussian(x, z, gamma):
    """
    Calculates the Radial Basis Function (RBF) with a Gaussian kernel.
    
    Parameters:
    x, z: numpy arrays of the same length representing two points in n-dimensional space
    gamma: a parameter of the RBF (must be greater than 0)
    
    Returns:
    The result of the RBF
    """
    distance = np.linalg.norm(x-z)**2
    return np.exp(-gamma * distance)


# Example usage:
x = np.array([1, 2, 3])
z = np.array([4, 5, 6])
gamma = 0.1


print(rbf_gaussian(x, z, gamma))
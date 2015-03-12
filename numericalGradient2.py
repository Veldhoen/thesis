import numpy as np

'''
Compute the numerical gradients, by determining the difference in error
when increasing/ decreasing the parameters by a small amount (epsilon)

Input:
 - network: a neural network
            NB: a function error(theta, target) must be implemented for the network
 - theta:   a numpy array of parameters
 - target:  the target value, if any, in the appropriate format for the error function

Output:
 - numgrad: a numpy array of same dimensionality as theta,
            with the numerical gradients of the parameters
'''
def numericalGradient(network,theta, target=None):
    epsilon = 0.0001
    numgrad = np.zeros_like(theta)
    for i in range(len(theta)):
        old = theta[i]
        theta[i] = old + epsilon
        errorPlus = network.error(theta,target)
        theta[i] = old - epsilon
        errorMin = network.error(theta,target)
        # reset theta[i]
        theta[i] = old
        numgrad[i] = (errorPlus-errorMin)/(2*epsilon)
    return numgrad
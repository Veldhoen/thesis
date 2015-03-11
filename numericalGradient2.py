import numpy as np
#import naturalLogic as nl

def numericalGradient(tree,theta, target):
#     global d,L
#     d = len(bc)
#     L = words
    epsilon = 0.0001

    # activate all nodes in the tree
    tree.forward(theta)

    # compute the numerical gradients
    numgrad = np.zeros_like(theta)
    for i in range(len(theta)):
        old = theta[i]
        theta[i] = old + epsilon
        errorPlus = tree.error(target,theta)
        theta[i] = old - epsilon
        errorMin = tree.error(target,theta)
        # reset theta[i]
        theta[i] = old
        numgrad[i] = (errorPlus-errorMin)/(2*epsilon)

        # decompose numerical and anlytical gradients
        # back into Wc, bc, Wr and br
    return numgrad
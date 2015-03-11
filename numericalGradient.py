import numpy as np

def numericalGradient(tree,Wc,bc,Wr,br,words):
    global d,L
    d = len(bc)
    L = words
    epsilon = 0.0001

    # activate all nodes in the tree
    tree.forwardPass(Wc,bc,Wr,br,L,True,True)
   # compute the analytical gradients
    gradWc,gradBc,gradWr,gradBr = tree.backprop(np.zeros(d),Wc,bc,Wr,br,L, True)
    grad = np.concatenate([np.reshape(gradWc,-1),gradBc,np.reshape(gradWr,-1),gradBr])
    theta = np.concatenate([np.reshape(W, -1) for W in [Wc,bc,Wr,br]])

    # compute the numerical gradients
    numgrad = np.zeros_like(theta)
    for i in range(len(theta)):
        old = theta[i]
        theta[i] = old + epsilon
        errorPlus = error(tree, theta)
        theta[i] = old - epsilon
        errorMin = error(tree, theta)
        # reset theta[i]
        theta[i] = old
        numgrad[i] = (errorPlus-errorMin)/(2*epsilon)

    if np.array_equal(numgrad,grad): print 'numerical and analytical gradients are equal.'
    else:
        print 'Difference numerical and analytical gradients:', np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
        # decompose numerical and anlytical gradients
        # back into Wc, bc, Wr and br
        left = 0
        right = left + d*2*d
        print 'Wc diff:', np.linalg.norm(numgrad[left:right]-grad[left:right])/np.linalg.norm(numgrad[left:right]+grad[left:right])
        left = right
        right = left + d
        print 'bc diff:', np.linalg.norm(numgrad[left:right]-grad[left:right])/np.linalg.norm(numgrad[left:right]+grad[left:right])
        left = right
        right = left + 2*d*d
        print 'Wr diff:', np.linalg.norm(numgrad[left:right]-grad[left:right])/np.linalg.norm(numgrad[left:right]+grad[left:right])
        left = right
        right = left + 2*d
        print 'br diff:', np.linalg.norm(numgrad[left:right]-grad[left:right])/np.linalg.norm(numgrad[left:right]+grad[left:right])

def error(tree, theta, verbose= False):
    # Retrieve Wc, bc, Wr, br from flat theta
    left = 0
    right = left + d*2*d
    WcP = np.reshape(theta[left:right],(d,2*d))
#    print 'Wc',left,right
    left = right
    right = left + d
    bcP = theta[left:right]
#    print 'bc',left,right
    left = right
    right = left + 2*d*d
    WrP = np.reshape(theta[left:right],(2*d,d))
#    print 'Wr',left,right
    left = right
    right = left + 2*d
    brP = theta[left:right]
#    print 'br',left,right
    # compute and return error
    return recurseError(tree,WcP,bcP,WrP,brP, verbose)

def recurseError(tree,WcP,bcP,WrP,brP, verbose = False):
    rootError = tree.reconstructionError(WcP,bcP,WrP,brP,L, verbose)
    return rootError + sum([recurseError(child,WcP,bcP,WrP,brP, verbose) for child in tree.children])

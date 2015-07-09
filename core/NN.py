from __future__ import division
class Node():
  def __init__(self,inputs, outputs, cat,nonlinearity):
#    print 'Node.init', cat, inputs
    self.inputs = inputs
    self.outputs = outputs
    self.cat = cat
    self.nonlin = nonlinearity

  def forward(self,theta, activateIn = True, activateOut = False):
    if activateIn: self.inputs.forward(theta)

    inputsignal = np.concatenate([c.a for c in inputs])

    M= theta[self.cat+'M']
    b= theta[self.cat+'B']

    self.z = M.dot(activationIn)+b
    self.a, self.ad = activate(self.z, self.activation)
    if activateOut: self.outputs.forward(theta, False, True)

  def backprop(self,theta, delta, gradient=None, addOut = False):
    if addOut: True #add a delta message from its outputs (e.g., reconstructions)


    if gradient is None: gradient = theta.zeros_like(self, sparseWords = True)
    inputsignal = np.concatenate([c.a for c in inputs])
    dinputsignal = np.concatenate([c.ad for c in inputs])

    deltaB =np.multiply(np.transpose(theta[self.cat+'IM']).dot(delta),dinputsignal)
    lens = [len(c.a) for c in inputs]
    splitter = [sum(lens[:i]) for i in range(len(lens))]

    deltaBits = np.split(deltaB,splitter[:-1])
    [inputNode.backprop(theta, delt, gradient) for inputnode,delt in zip(self.inputs,deltaBits)]



    inputSignal= np.concatenate([child.innerA for child in self.children])
    childrenads = np.concatenate([child.innerAd for child in self.children])
    # compute delta to backprop and backprop it
    deltaB = np.split(np.multiply(np.transpose(theta[self.cat+'IM']).dot(delta),childrenads),len(self.children))
    [self.children[i].backpropInner(deltaB[i], theta, gradients) for i in xrange(len(self.children))]
    # update gradients for this node
    gradients[self.cat+'IM']+= np.outer(delta,childrenas)
    gradients[self.cat+'IB']+=delta

  def __str__(self):
    return '['+self.cat[1]+' '+ ' '.join([str(child) for child in self.inputs])+']'
    #return '['+self.cat[0]+' '+ ' '.join([str(child) for child in self.inputs])+']'


class Leaf(Node):
  def __init__(self,outputs,cat='word', word='', index=0,nonlinearity='identity'):
    Node.__init__(self,[], outputs, cat,nonlinearity)
    self.word = word
    self.index = index

  def forward(self,theta, activateIn = True, activateOut = False):
    True
    # look up

  def __str__(self):
    return self.word#+'('+self.cat+')'
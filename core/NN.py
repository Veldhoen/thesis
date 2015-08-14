from __future__ import division
import activation
import numpy as np
import sys

import myTheta

class Node():
  def __init__(self,inputs, outputs, cat,nonlinearity):
#    print 'Node.init', cat, inputs
    self.inputs = inputs
    self.outputs = outputs
    self.cat = cat
    self.nonlin = nonlinearity

  def forward(self,theta, activateIn = True, activateOut = False, inputSet=False):
#    print 'forward',theta[('composition', '#X#', '(#X#, #X#)', 'I', 'M')][0][:3]
    if activateIn:
      [i.forward(theta, activateIn,activateOut) for i in self.inputs]
    if not inputSet:
      self.inputsignal = np.concatenate([c.a for c in self.inputs])
      self.dinputsignal = np.concatenate([c.ad for c in self.inputs])
    M= theta[self.cat+('M',)]
    b= theta[self.cat+('B',)]
    if M is None or b is None:
      print 'Fail to forward node, no matrix and bias vector:', self.cat
      sys.exit()
#    try:
    self.z = M.dot(self.inputsignal)+b
    self.a, self.ad = activation.activate(self.z, self.nonlin)
 #    except:
#       print self.cat, self.inputsignal.shape, M.shape, b.shape
#       self.z = M.dot(self.inputsignal)+b
#       self.a, self.ad = activation.activate(self.z, self.nonlin)
    if activateOut:
#       lens = [int(np.shape(theta[c.cat+('M',)])[1]) for c in self.outputs] #but they are not activated yet!
#       splitter = [sum(lens[:i]) for i in range(len(lens))][1:]
#       for node, a, ad in zip(self.outputs, np.split(self.a,splitter), np.split(self.ad,splitter)):
#         node.forward(a,ad,theta)

      for node in self.outputs:
        node.forward(theta, False,activateOut)
#         elif type(node) == Leaf: print 'a leaf cannot be an output of another node!'
#         else:
#           print type(node), self.a.shape
#           node.forward(self.a,self.ad, theta)

#       if len(self.outputs)==1:
# #        assert isinstance(self.outputs[0],myRAE.Reconstruction)
#         self.outputs[0]
#       else: print 'forwarding to too many outputs!'


    #  print 'do forward outputs'
   #   [i.forward(theta, activateIn,activateOut) for i in self.outputs] #self.outputs.forward(theta, activateIn,activateOut)

  def backprop(self,theta, delta, gradient, addOut = False, moveOn=True):
#    if self.a.shape != delta.shape:
#    print 'backprop Node,',self.cat, 'a:', self.a.shape,'delta:', delta.shape, 'input:', self.inputsignal.shape
#    print'\t theta',type(theta),'gradients',type(gradient)

    if addOut: #add a delta message from its outputs (e.g., reconstructions)
      print 'addOut'
      delta += np.concatenate([out.backprop(theta, gradient) for out in self.outputs])

    M= theta[self.cat+('M',)]
#    grm =
    gradient[self.cat+('M',)]+= np.outer(delta,self.inputsignal)
    gradient[self.cat+('B',)]+=delta
    # except:
#       print 'backprop fails for', self.cat, 'delta:', delta.shape, 'input:', self.inputsignal.shape
#       sys.exit()

    deltaB =np.multiply(np.transpose(M).dot(delta),self.dinputsignal)
    if moveOn:
      lens = [len(c.a) for c in self.inputs]
      splitter = [sum(lens[:i]) for i in range(len(lens))][1:]
      [inputNode.backprop(theta, delt, gradient) for inputNode,delt in zip(self.inputs,np.split(deltaB,splitter))]
    else:
      return deltaB


  def __str__(self):
    if self.cat[-1]=='I': return '('+self.cat[1]+' '+ ' '.join([str(child) for child in self.inputs])+')'
    if self.cat[-1]=='O': return '['+self.cat[1]+' '+ ' '.join([str(child) for child in self.inputs])+']' #'&'.join([str(c) for c in self.inputs])
    else: return '<'+self.cat[0]+' '+ ' '.join([str(child) for child in self.inputs])+'>'
    #return '['+self.cat[0]+' '+ ' '.join([str(child) for child in self.inputs])+']'


class Leaf(Node):
  def __init__(self,outputs,cat, key=0,nonlinearity='identity'):
    Node.__init__(self,[], outputs, cat,nonlinearity)
    self.key = key

  def forward(self,theta, activateIn = True, activateOut = False):
#    print 'forward leaf', self.cat, self.key, type(self.key)
    try: self.z = theta[self.cat][self.key]
#     try: self.z = theta[self.cat][self.key]
    except:
      print 'Fail to forward Leaf:', self.cat, self.key
      sys.exit()

    self.a, self.ad = activation.activate(self.z,self.nonlin)
    if activateOut:
      [i.forward(theta, False,activateOut) for i in self.outputs] #self.outputs.forward(theta, activateIn,activateOut)

  def backprop(self,theta, delta, gradient, addOut = False):
    gradient[self.cat][self.key] += delta



  def __str__(self):
    return str(self.key)#+'('+self.cat+')'
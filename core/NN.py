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

  def forward(self,theta, activateIn = True, activateOut = False):
#    print 'forward',self.cat[0], self.cat[-1] , self
    if activateIn:
  #    print 'do forward inputs'
      [i.forward(theta, activateIn,activateOut) for i in self.inputs]

    inputsignal = np.concatenate([c.a for c in self.inputs])
#    print 'forward',self.cat[0], self.cat[-1] , len(inputsignal)


    M= theta[self.cat+('M',)]
    b= theta[self.cat+('B',)]
    if M is None or b is None:
      print 'Fail to forward node, no matrix and bias vector:', self.cat
      sys.exit()

    try: self.z = M.dot(inputsignal)+b
    except: 
      print 'Unable to forward', self.cat
      print 'inputs:', self.inputs, len(inputsignal)
      print 'shape M:', np.shape(M)
      sys.exit()


    self.a, self.ad = activation.activate(self.z, self.nonlin)
    if activateOut:
    #  print 'do forward outputs'
      [i.forward(theta, activateIn,activateOut) for i in self.outputs] #self.outputs.forward(theta, activateIn,activateOut)

  def backprop(self,theta, delta, gradient, addOut = False):
#    print 'backprop Node,',delta, self.cat#[0], self#,'types:'
#    print'\t theta',type(theta),'gradients',type(gradient)

    if addOut: True #add a delta message from its outputs (e.g., reconstructions)
#    if gradient is None: gradient = myTheta.Gradient(theta) #theta.zeros_like(sparseWords = True)

    inputsignal = np.concatenate([c.a for c in self.inputs])
    dinputsignal = np.concatenate([c.ad for c in self.inputs])

    M= theta[self.cat+('M',)]
    # something goes wrong in this multiplication:
    # for the u node, the result is a square (15x15) instead of a vector (1x15).
    # Solved by backproping delta[0] at the start (messy solution)
    deltaB =np.multiply(np.transpose(M).dot(delta),dinputsignal)
#    print 'M:', np.shape(M), 'deltaB',np.shape(deltaB)
    lens = [len(c.a) for c in self.inputs]

    splitter = [sum(lens[:i]) for i in range(len(lens))][1:]
    [inputNode.backprop(theta, delt, gradient) for inputNode,delt in zip(self.inputs,np.split(deltaB,splitter))]

    gradient[self.cat+('M',)]+= np.outer(delta,inputsignal)
    gradient[self.cat+('B',)]+=delta

  def __str__(self):
    if self.cat[-1]=='I': return '('+self.cat[1]+' '+ ' '.join([str(child) for child in self.inputs])+')'
    if self.cat[-1]=='O': return '['+self.cat[1]+' '+ ' '.join([str(child) for child in self.inputs])+']' #'&'.join([str(c) for c in self.inputs])
    else: return '<'+self.cat[0]+' '+ ' '.join([str(child) for child in self.inputs])+'>'
    #return '['+self.cat[0]+' '+ ' '.join([str(child) for child in self.inputs])+']'


class Leaf(Node):
  def __init__(self,outputs,cat, key='',nonlinearity='identity'):
    Node.__init__(self,[], outputs, cat,nonlinearity)
    self.key = key

  def forward(self,theta, activateIn = True, activateOut = False):
#    print 'forward leaf',self.cat[0], self

    try: index = theta.lookup[self.cat].index(self.key)
    except: index = theta.lookup[self.cat].index('UNKNOWN')

    try: self.z = theta[self.cat][index]
    except:
      print 'Fail to forward Leaf:', self.cat, self.key
      sys.exit()

    self.a, self.ad = activation.activate(self.z,self.nonlin)
    if activateOut:
#      print 'do forward outputs'
      [i.forward(theta, activateIn,activateOut) for i in self.outputs] #self.outputs.forward(theta, activateIn,activateOut)

  def backprop(self,theta, delta, gradient, addOut = False):
#    print 'backprop Leaf,',self.cat, self#,' types:'
    try: index = theta.lookup[self.cat].index(self.key)
    except: index = theta.lookup[self.cat].index('UNKNOWN')
    words = gradient[self.cat]
    words[index] += delta



  def __str__(self):
    return self.key#+'('+self.cat+')'
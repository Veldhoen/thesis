import NN
from __future__ import division
import numpy as np
import sys


class RAENode(NN.Node):
  def __init__(self,children,cat, nonlinearity):
    NN.Node.__init__(self.children,cat,nonlinearity)
    self.reconstructions = [RAENode(s
    if len(self.children) > 0:
      self.reconstruction = Reconstruction(self)

class Reconstruction():
  def __init__(self,original):
    self.original = original
    self.children = [Reconstruction(child) for child in original.children]

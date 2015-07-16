class NetworkBank():
  def __init__(treebanks,vocabulary):
    for name in treebanks:
      with open(name,'rb')

    self.files = fromFiles
    self.m = 0
    self.examples = None
  def getNWs():
    if len(self.files)==1 and self.examples is not None: return self.examples

    with open(fromFile,'rb') as f:
      self.examples = pickle.load(f)
    for kind, trees in examples.iteritems():
      for i in xrange(len(trees)):
        self.examples[kind][i] = naturalLogic.iornnFromTree(trees[i][0][0], vocabulary,grammar)
    self.updateM()
    return self.examples

  def updateM():
    self.m+=1
    if self.m>=len(self.files):
      random.shuffle(self.files)
      m = 0

def beginSmall(treebank):
  usedTreeF

  m = 0
  if len(treeFiles)==1:    examples = getData(treeFiles[m])


#  Assume theta is initialized with a single, general composition function


# train for one epoch (or stop early). Skip every sentence longer than x words
# increase x a few times

  if len(treeFiles)>1:
    examples = getData(treeFiles[m])
    m +=1
    if m==len(treeFiles):
      random.shuffle(treeFiles)
      m = 0

  epoch(theta, examples, historical_grad = None, earlyStopping = False):


# replace composition function specialized for LHS, initialize with general one
# train for one epoch (or stop early)

# replace composition function specialized for rule, initialize with LHS one
# train for one epoch (or stop early)









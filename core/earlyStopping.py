


def stopNow(trainLoss, validLoss):
  k = 10
  if len(trainLoss)<k or len(validLoss)<k: return False

  alpha = 0.75
  gl = GL(validLoss)
  pk = Pk(trainLoss, k)
  PQ = gl/pk
  print 'Stopping criterion. GL:',gl,'Pk:',pk,'PQ:', PQ
  return PQ>alpha

# generalization loss
def GL(validLoss):
  return validLoss[-1]/min(validLoss)-1

# training progress
def Pk(trainLoss,k):
  k=min(k,len(trainLoss))
  return sum(trainLoss[-k:])/(k*min(trainLoss[-k:]))-1




def stopNow(trainLoss, validLoss):
  if len(trainLoss)<1 or len(validLoss)<1: return False
  k = 10
  alpha = 1
  PQ = GL(validLoss)/Pk(trainLoss, k)
  print 'stopping criterion PQ:', PQ
  return PQ>alpha

# generalization loss
def GL(validLoss):
  return validLoss[-1]/min(validLoss)-1

# training progress
def Pk(trainLoss,k):
  return sum(trainLoss[-k:])/(k*min(trainLoss[-k:]))-1

from IORNN import *



def main():
  source = 'data/treesFlickr.pik'
  with open(source, 'rb') as f:
    nws, vocabulary = pickle.load(f)



def evaluateNW(nw,theta):
  nwords = len(theta['wordIM'])
  
  for leaf in nw:
    scores = np.zeros(nwords)
    for x in range(nwords):
      scores[x] = nw.score(theta,x)
      rank = scores.argsort().argsort()[leaf.index]

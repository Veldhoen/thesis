def gradientCheck(theta, network, target):
  network.activateNW(theta)

  # compute analyticial and numerical gradient
  grad = network.backprop(theta, target)
  numgrad = network.numericalGradient(theta,target)

  # flatten gradient objects and report difference
  gradflat = np.array([])
  numgradflat = np.array([])
  for name in theta.dtype.names:
    ngr = np.reshape(numgrad[name],-1)
    gr = np.reshape(grad[name],-1)
    if np.array_equal(gr,ngr): diff = 0
    else: diff = np.linalg.norm(ngr-gr)/(np.linalg.norm(ngr)+np.linalg.norm(gr))
    print 'Difference '+name+' :', diff
    gradflat = np.append(gradflat,gr)
    numgradflat = np.append(numgradflat,ngr)
  print 'Difference overall:', np.linalg.norm(numgradflat-gradflat)/(np.linalg.norm(numgradflat)+np.linalg.norm(gradflat))






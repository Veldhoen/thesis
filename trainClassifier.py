import core.classifier as cl
import sys,os

def main(thetaFile):
  if not os.path.isfile(thetaFile): 
    print 'no file containing theta:', thetaFile
    sys.exit()
  cl.classifyInference(thetaFile)

if __name__ == "__main__": main(sys.argv[1])
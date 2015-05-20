experiment=$1
data='data/flickr'
treePost='TREES.pik'
vocPost='VOC.pik'
out="models/flickr$experiment.pik"
emb='data/senna.pik'
nEpochs=5
bSize=100
alpha='0.2'
lambda='0.0005'
cores='5'

python -u trainIOCopy.py \
  -t $data$treePost \
  -v $data$vocPost \
  -o $out \
  -dwrd 5 \
  -n $nEpochs \
  -b $bSize \
  -a $alpha \
  -l $lambda \
  -c $cores
echo "$experiment $!" >> psIds.txt



#  -e $emb \
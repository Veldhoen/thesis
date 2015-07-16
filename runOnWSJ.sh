experiment=$1
data='data/WSJ'
treePost='TREES.pik'
vocPost='VOC.pik'
gramPost='RULES.pik'
out="models/WSJ$experiment"
nEpochs=5
bSize=10
alpha='0.05'
# lambda='0.1'
cores=1

echo "$experiment $$" >> psIds.txt

for lambda in 0
#1.0 0.5 0.1 0.01 0
do
#  myNohup=nohups/$experiment-$lambda.nohup.out
  echo "training on $data for $nEpochs epochs"
#  nohup python -u -W once \
  python -u -W once \
    trainIOCopy.py \
    -exp $experiment \
    -s $data \
    -g Rules 25 \
    -o $out \
    -dwrd 5 \
    -n $nEpochs \
    -b $bSize \
    -a $alpha \
    -l $lambda \
    -c $cores \
#  >> $myNohup &
#  echo "python main $!" >> psIds.txt
done






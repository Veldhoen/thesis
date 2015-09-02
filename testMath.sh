coredir=testMath


rm -r $coredir/*

for emb in orth rand
do
  for kind in RAE IORNN
  do
    for spec in wS nS
    do
      for fix in tE fE
      do
        for $comp in simple complex
        do
          echo $spec $fix
          thisdir=$coredir/$comp/$kind/$spec/$fix/$emb
          mkdir -p $thisdir
          nohup python generateMathExamples.py $thisdir $spec $fix $kind $emb $comp\
          > $coredir/$kind-$spec-$fix.nohup.out &
          echo "$comp/$kind/$spec/$fix/$emb $!"
        done
      done
    done
  done
done
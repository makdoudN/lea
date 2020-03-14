seed=$(( ( RANDOM % 10000 )  + 1 ))
EPOCHS=200
EXPERTS=${1}
EXPERTS_SEED=993

for A in PR
do
  for E in HalfCheetah Walker2d
  do
    for TASK in GravityHalf GravityOneAndHalf SmallFoot SmallLeg
    do
      for nC in 1 5 10 25 50
      do
        for pR in 0.0 0.1 0.2 0.5 0.75 1
        do
          tsp python ../../../train.py env.name=${E}${TASK}-v1 rl=${A} \
            num_epochs=${EPOCHS} seed=${seed} experiment=MJ2MJT.Gravity/${A}_${nC}_${pR}/${E}/${TASK} \
            experts_path=${EXPERTS}/${E}-v3/${EXPERTS_SEED} rl.kwargs.pi_reuse=${pR} rl.kwargs.num_commit=${nC}
        done
      done
    done
  done
done



seed=$(( ( RANDOM % 10000 )  + 1 ))
EPOCHS=200
EXPERTS=${1}
EXPERTS_SEED=993

for A in RPL RPL-TD3
do
  for E in HalfCheetah Hopper Walker2d
  do
    for TASK in GravityHalf GravityThreeQuarters GravityOneAndQuarter GravityOneAndHalf SmallFoot SmallLeg SmallTorso SmallThigh BigFoot BigThigh BigTorso BigLeg
    do
      tsp python ../../train.py env.name=${E}${TASK}-v1 rl=${A} \
          num_epochs=${EPOCHS} seed=${seed} experiment=MJ2MJT.Gravity/${A}/${E}/${TASK} \
          experts_path=${EXPERTS}/${E}-v3/${EXPERTS_SEED}
    done
  done
done



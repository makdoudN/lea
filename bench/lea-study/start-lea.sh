seed=$(( ( RANDOM % 10000 )  + 1 ))
EPOCHS=200
EXPERTS=${1}
EXPERTS_SEED=993
# NOTE ADD LEA-TD3 in the future
A=LEA

for E in HalfCheetah Hopper Walker2d
do
  for TASK in GravityHalf GravityThreeQuarters GravityOneAndQuarter GravityOneAndHalf SmallFoot SmallLeg SmallTorso SmallThigh BigFoot BigThigh BigTorso BigLeg
  do
    for NC in 1 5 10
    do
    tsp python ../../train.py env.name=${E}${TASK}-v1 rl=${A} \
        num_epochs=${EPOCHS} seed=${seed} experiment=LEA-NC_${NC}-reuse_True/MJ2MJT/${A}/${E}/${TASK}/ \
        experts_path=${EXPERTS}/${E}-v3/${EXPERTS_SEED} rl.kwargs.reuse=true rl.kwargs.pi_reuse=0.2 rl.kwargs.num_commit=${NC} \
        rl.kwargs.use_q_guidance=true rl.kwargs.use_pi_guidance=false
    done
  done
done



seed=$(( ( RANDOM % 10000 )  + 1 ))
EPOCHS=200
EXPERTS=${1}
EXPERTS_SEED=993
A=LEA-TD3


for E in HalfCheetah Hopper Walker2d
do
  for TASK in GravityHalf GravityThreeQuarters GravityOneAndQuarter GravityOneAndHalf SmallFoot SmallLeg SmallTorso SmallThigh BigFoot BigThigh BigTorso BigLeg
  do
    for NC in 1 5 10
    do

    tsp python ../../train.py env.name=${E}${TASK}-v1 rl=${A} \
        num_epochs=${EPOCHS} seed=${seed} experiment=${A}_${NC}_QGuidance_Reuse/${E}/${TASK}/ \
        experts_path=${EXPERTS}/${E}-v3/${EXPERTS_SEED} rl.kwargs.reuse=true rl.kwargs.pi_reuse=0.2 rl.kwargs.num_commit=${NC} \
        rl.kwargs.use_q_guidance=true rl.kwargs.use_pi_guidance=false

    tsp python ../../train.py env.name=${E}${TASK}-v1 rl=${A} \
        num_epochs=${EPOCHS} seed=${seed} experiment=${A}_${NC}_NoQGuidance_Reuse/${E}/${TASK}/ \
        experts_path=${EXPERTS}/${E}-v3/${EXPERTS_SEED} rl.kwargs.reuse=true rl.kwargs.pi_reuse=0.2 rl.kwargs.num_commit=${NC} \
        rl.kwargs.use_q_guidance=false rl.kwargs.use_pi_guidance=false

    tsp python ../../train.py env.name=${E}${TASK}-v1 rl=${A} \
        num_epochs=${EPOCHS} seed=${seed} experiment=${A}_${NC}_QGuidance_NoReuse/${E}/${TASK}/ \
        experts_path=${EXPERTS}/${E}-v3/${EXPERTS_SEED} rl.kwargs.reuse=false rl.kwargs.pi_reuse=0.0 rl.kwargs.num_commit=${NC} \
        rl.kwargs.use_q_guidance=false rl.kwargs.use_pi_guidance=false

    tsp python ../../train.py env.name=${E}${TASK}-v1 rl=${A} \
        num_epochs=${EPOCHS} seed=${seed} experiment=${A}_${NC}_NoQGuidance_NoReuse/${E}/${TASK}/ \
        experts_path=${EXPERTS}/${E}-v3/${EXPERTS_SEED} rl.kwargs.reuse=false rl.kwargs.pi_reuse=0.0 rl.kwargs.num_commit=${NC} \
        rl.kwargs.use_q_guidance=false rl.kwargs.use_pi_guidance=false

    done
  done
done



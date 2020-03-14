seed=$(( ( RANDOM % 10000 )  + 1 ))
EPOCHS=200
EXPERTS=${1}
EXPERTS_SEED=993
A=LEA-v2

for E in HalfCheetah Walker2d
do
  for TASK in GravityHalf GravityOneAndHalf SmallFoot BigLeg
  do
    for NC in 1 5
    do
      for LAM in 0 0.1 0.01 0.001 0.0001 1 10
      do
      tsp python ../../train.py env.name=${E}${TASK}-v1 rl=${A} \
          num_epochs=${EPOCHS} seed=${seed} experiment=LEA-NC_${NC}-reuse_True/MJ2MJT/${A}/${E}/${TASK}/ \
          experts_path=${EXPERTS}/${E}-v3/${EXPERTS_SEED} rl.kwargs.reuse=true rl.kwargs.pi_reuse=0.2 rl.kwargs.num_commit=${NC} \
          rl.kwargs.use_q_guidance=true rl.kwargs.use_pi_guidance=false rl.kwargs.lam_aux_policy_guidance=${LAM} rl.kwargs.aux_policy_guidance_version=0
      done
    done
  done
done



export TF_CPP_MIN_LOG_LEVEL=3
export DTENSOR_JOB_NAME=worker
export DTENSOR_JOBS=localhost:9890,localhost:9891,localhost:9892,localhost:9893
export DTENSOR_NUM_CLIENTS=4

for i in 0 1 2 3; do
(
export CUDA_VISIBLE_DEVICES=${i}
export DTENSOR_CLIENT_ID=${i}
exec $*
) &
done
wait

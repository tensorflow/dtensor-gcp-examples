HOSTNAME=`hostname`

export TF_CPP_MIN_LOG_LEVEL=3
export DTENSOR_JOB_NAME=worker
export DTENSOR_JOBS=`cat dtensor-jobs`
export DTENSOR_NUM_CLIENTS=`python get-clients.py`

for i in `python get-clients.py`; do
  export DTENSOR_CLIENTS_ID="$i"
  python dtensor-client.py
done

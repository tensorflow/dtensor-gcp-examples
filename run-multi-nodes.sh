HOSTNAME=`hostname`
DTENSOR_JOBS_FILE=/etc/dtensor-jobs
if ! [[ -f "${DTENSOR_JOBS_FILE}" ]]; then
  echo "${DTENSOR_JOBS_FILE}" is missing. Ensure the cluster is bootstraped.
fi

export TF_CPP_MIN_LOG_LEVEL=3
export DTENSOR_JOB_NAME=worker
export DTENSOR_JOBS=`cat ${DTENSOR_JOBS_FILE}`
export DTENSOR_NUM_CLIENTS=`python get-clients.py`

for i in `python get-clients.py`; do
  export DTENSOR_CLIENTS_ID="$i"
  python dtensor-client.py
done

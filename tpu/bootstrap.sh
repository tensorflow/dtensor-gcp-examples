#! bin/bash
# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configures a single-client v2-8 TPU VM for DTensor.

# This command produces cluster-run.sh, which can be used to run command
# on the TPU VM.
#
# The git repo is cloned to the VMs.

if [ $# -ne 1 ]; then
  echo 1>&2 "$0: need to specify a TPU topology. e.g. v2-8"
  exit 2
fi

TOPOLOGY=$1
NAME="${USER}-dtensor-tpu-${TOPOLOGY}-test"
ZONE=europe-west4-a
VERSION="v2-nightly"
NUM_CORES=$(cut -d "-" -f2- <<< $TOPOLOGY)
export NUM_WORKERS=$(($NUM_CORES / 8))
export GCS_BUCKET=${GCS_BUCKET:-dtensor-checkpoints}

if [ $NUM_WORKERS -gt 1 ]; then
  VERSION+="-pod"
fi

bash `dirname $0`/make-cluster-commands.sh "${NAME}" "${ZONE}" "${NUM_WORKERS}"

set -x
gcloud compute tpus tpu-vm create $NAME  \
    --zone=$ZONE  \
    --accelerator-type=$TOPOLOGY  \
    --version=$VERSION

gcloud compute firewall-rules create allow-ssh --allow=tcp:22
set +x

while bash cluster-run.sh ls |grep 'exited with return code'; do
  echo Health checking
  sleep 10
done

bash cluster-bcast.sh requirements.txt ./
bash cluster-run.sh pip3 install -r requirements.txt

bash cluster-run.sh "if ! [[ -d dtensor-gcp-examples ]]; then git clone https://github.com/tensorflow/dtensor-gcp-examples; fi"
bash cluster-run.sh "cd dtensor-gcp-examples; git pull;"
bash cluster-run.sh "ls -l dtensor-gcp-examples;"

cat > launch <<EOF
#! /bin/bash

export TF_CPP_MIN_LOG_LEVEL=3
export DTENSOR_CLIENT_ID="\$(echo \$HOSTNAME | rev | cut -d "-" -f1 | rev)"
export DTENSOR_NUM_CLIENTS=$NUM_WORKERS
if [ \$DTENSOR_NUM_CLIENTS -gt 1 ]; then
  export DTENSOR_JOB_NAME=worker
fi
WORKER_NAME="\$(echo \$HOSTNAME | rev | cut -d "-" -f1 --complement | rev)"
PORT=19011
export DTENSOR_JOBS="\${WORKER_NAME}-0:\${PORT}"

for ((i=1;i<$NUM_WORKERS;i++)); do
  DTENSOR_JOBS+=",\${WORKER_NAME}-\$i:\${PORT}"
done

(exec \$*) &
wait
EOF
chmod +x launch

bash cluster-bcast.sh launch ./

cat <<EOF
Next, run the application with,

  bash cluster-run.sh "./launch python3 dtensor-gcp-examples/dtensor-app-naive.py --ckpt_path_prefix=gs://${GCS_BUCKET}/app_naive --device_type=TPU"

  bash cluster-run.sh "./launch python3 dtensor-gcp-examples/dtensor-keras-bert.py --ckpt_path_prefix=gs://${GCS_BUCKET}/keras_bert  --device_type=TPU"

When done, delete the cluster with,

  bash cluster-delete.sh
EOF

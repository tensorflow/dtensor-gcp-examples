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

#
# Configures a GPU VM cluster for DTensor.

# This command produces cluster-run.sh, which can be used to run command
# on all nodes of the cluster.
#
# The git repo is cloned to the VMs.

DEVICE_TYPE=${1:-gpu}

export GCS_BUCKET=${GCS_BUCKET:-dtensor-checkpoints}
export IMAGE_FAMILY=common-cu113
export ZONE=us-west1-b
export INSTANCE_TYPE="n1-standard-8"
export COUNT=4
export NAME_PREFIX="${USER}-dtensor-cluster-${DEVICE_TYPE}"
export PORT=9898
export TAGS="${NAME_PREFIX}-node"

case "${DEVICE_TYPE}" in
  "dev")
  export ACCELERATOR="type=nvidia-tesla-k80,count=2"
  ;;
  "cpu")
  export ACCELERATOR=""
  ;;
  "gpu" )
  export ACCELERATOR="type=nvidia-tesla-v100,count=2"
  ;;
  "*" )
  echo "Unknown device type ${DEVICE_TYPE}"
  exit 1
  ;;
esac

INSTANCES=()
for i in `seq -s " " -f %03g 1 $COUNT`; do
  INSTANCES+=("${NAME_PREFIX}-${i}")
done

bash `dirname $0`/../make-cluster-commands.sh "${NAME_PREFIX}" "${ZONE}" "${INSTANCES[@]}"

set -x
gcloud compute instances bulk create --predefined-names=$(printf "%s," ${INSTANCES[@]} | sed 's/,$//') \
     --zone=$ZONE    \
     --image-family=$IMAGE_FAMILY     \
     --image-project=deeplearning-platform-release   \
     --on-host-maintenance=TERMINATE   \
     --accelerator="$ACCELERATOR"    \
     --machine-type=$INSTANCE_TYPE     \
     --boot-disk-size=120GB   \
     --boot-disk-type=pd-ssd   \
     --scopes=default,storage-rw \
     --metadata="install-nvidia-driver=True"  \
     --count=${COUNT} \
     --tags=${TAGS}
set +x

while bash cluster-run.sh ls |grep 'exited with return code'; do
  echo Health checking
  sleep 10
done

# Creates a per machine launcher script.
DTENSOR_JOBS_LIST=$(
for i in ${INSTANCES[@]}; do
  echo -n "${i}:${PORT}\\n"
done
)
echo "$DTENSOR_JOBS_LIST"
sed "/^@DTENSOR_JOBS_LIST@$/c${DTENSOR_JOBS_LIST}" launch.py.in > launch
chmod +x launch

bash cluster-run.sh "sudo /opt/conda/bin/conda clean -q -y --all"
bash cluster-run.sh "conda create -q -y -n py310 python=3.10"
bash cluster-run.sh "conda activate py310; pip install -q tf-nightly tf-models-nightly"

bash cluster-bcast.sh launch ./
bash cluster-run.sh "if ! [[ -d dtensor-gcp-examples ]]; then git clone https://github.com/tensorflow/dtensor-gcp-examples; fi"
bash cluster-run.sh "cd dtensor-gcp-examples; git pull"
bash cluster-run.sh "ls -l dtensor-gcp-examples;"

cat <<EOF
Next, run the application with,

  bash cluster-run.sh "conda activate py310; ./launch python dtensor-gcp-examples/dtensor-app-naive.py --prefix=gs://${GCS_BUCKET}"

  bash cluster-run.sh "conda activate py310; ./launch python dtensor-gcp-examples/dtensor-keras-bert.py --prefix=gs://${GCS_BUCKET}"

When done, delete the cluster with,

  bash cluster-delete.sh
EOF

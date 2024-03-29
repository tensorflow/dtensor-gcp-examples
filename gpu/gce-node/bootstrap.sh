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
# Configures a GPU VM node for DTensor.

# This command produces cluster-run.sh, which can be used to run command
# on the node.
#
# The git repo is cloned to the VMs.

DEVICE_TYPE=${1:-gpu}

export GCS_BUCKET=${GCS_BUCKET:-dtensor-checkpoints}
export IMAGE_FAMILY=common-cu113
export ZONE=us-west1-b
export INSTANCE_TYPE="n1-standard-64"
export NAME="${USER}-dtensor-singlenode-${DEVICE_TYPE}"
export PORT=9898

case "${DEVICE_TYPE}" in
  "dev")
  export ACCELERATOR="type=nvidia-tesla-k80,count=8"
  ;;
  "cpu")
  export ACCELERATOR=""
  ;;
  "gpu" )
  export ACCELERATOR="type=nvidia-tesla-v100,count=8"
  ;;
  "*" )
  echo "Unknown device type ${DEVICE_TYPE}"
  exit 1
  ;;
esac

INSTANCES=($NAME)

bash `dirname $0`/../make-cluster-commands.sh "${NAME}" "${ZONE}" "${INSTANCES[@]}"

set -x
gcloud compute instances create $NAME \
     --zone=$ZONE    \
     --image-family=$IMAGE_FAMILY     \
     --image-project=deeplearning-platform-release   \
     --maintenance-policy=TERMINATE   \
     --accelerator="$ACCELERATOR"    \
     --machine-type=$INSTANCE_TYPE     \
     --boot-disk-size=200GB   \
     --boot-disk-type=pd-ssd   \
     --scopes=default,storage-rw \
     --metadata="install-nvidia-driver=True"
set +x

while bash cluster-run.sh ls |grep 'exited with return code'; do
  echo Health checking
  sleep 10
done

bash cluster-run.sh "sudo /opt/conda/bin/conda clean -q -y --all"
bash cluster-run.sh "conda create -q -y -n py3 python=3.8"
bash cluster-bcast.sh requirements.txt ./
bash cluster-run.sh "conda activate py3; pip install -r requirements.txt"
bash cluster-bcast.sh launch ./
bash cluster-run.sh "if ! [[ -d dtensor-gcp-examples ]]; then git clone https://github.com/tensorflow/dtensor-gcp-examples; fi"
bash cluster-run.sh "cd dtensor-gcp-examples; git pull;"
bash cluster-run.sh "ls -l dtensor-gcp-examples;"

cat <<EOF
Next, run the application with 4 clients:

  bash cluster-run.sh "conda activate py3; ./launch python dtensor-gcp-examples/dtensor_app_naive.py --device_type=GPU --ckpt_path_prefix=gs://${GCS_BUCKET}/app_naive"

  bash cluster-run.sh "conda activate py3; ./launch python dtensor-gcp-examples/dtensor-keras-bert.py --device_type=GPU --ckpt_path_prefix=gs://${GCS_BUCKET}/keras_bert"

As there only 1 node, you can also run the application with a single client.

  bash cluster-run.sh "conda activate py3; python dtensor-gcp-examples/dtensor_app_naive.py --device_type=GPU --ckpt_path_prefix=gs://${GCS_BUCKET}/app_naive"

  bash cluster-run.sh "conda activate py3; python dtensor-gcp-examples/dtensor-keras-bert.py --device_type=GPU --ckpt_path_prefix=gs://${GCS_BUCKET}/keras_bert"

When done, delete the cluster with,

  bash cluster-delete.sh
EOF

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

export GCS_BUCKET=${GCS_BUCKET:-dtensor-checkpoints}
export IMAGE_FAMILY=common-cu113
export ZONE=us-west1-b
export INSTANCE_TYPE="n1-standard-16"
export NAME="dtensor-singlenode"
export PORT=9898
export NUM_GPUS=8

INSTANCES=($NAME)

bash `dirname $0`/../make-cluster-commands.sh "${ZONE}" "${INSTANCES[@]}"

gcloud compute instances create $NAME \
     --zone=$ZONE    \
     --image-family=$IMAGE_FAMILY     \
     --image-project=deeplearning-platform-release   \
     --maintenance-policy=TERMINATE   \
     --accelerator="type=nvidia-tesla-v100,count=${NUM_GPUS}"    \
     --machine-type=$INSTANCE_TYPE     \
     --boot-disk-size=120GB   \
     --scopes=default,storage-rw \
     --metadata="install-nvidia-driver=True"  \

while bash cluster-run.sh ls |grep 'exited with return code'; do
  echo Health checking
  sleep 10
done

bash cluster-run.sh "sudo /opt/conda/bin/conda clean -q -y --all"
bash cluster-run.sh "conda create -q -y -n py310 python=3.10"
bash cluster-run.sh "conda activate py310; pip install -q tf-nightly tf-models-nightly"
# Upgrade to a tf-model-nightly version with our fixes.
bash cluster-run.sh "conda activate py310; pip install -q --no-deps --force tf-models-nightly==2.9.0.dev20220523 opencv-python-headless==4.5.4.60"
bash cluster-bcast.sh launch ./
bash cluster-run.sh "if ! [[ -d dtensor-gpu-gcp ]]; then git clone https://github.com/rainwoodman/dtensor-gpu-gcp; fi"
bash cluster-run.sh "cd dtensor-gpu-gcp; git pull"
bash cluster-run.sh "ls -l dtensor-gpu-gcp;"

cat <<EOF
Next, run the application with 4 clients:

  bash cluster-run.sh "conda activate py310; ./launch python dtensor-gpu-gcp/dtensor-app-naive.py --prefix=gs://${GCS_BUCKET}"

  bash cluster-run.sh "conda activate py310; ./launch python dtensor-gpu-gcp/dtensor-keras-bert.py --prefix=gs://${GCS_BUCKET}"

As there only 1 node, you can also run the application with a single client.

  bash cluster-run.sh "conda activate py310; python dtensor-gpu-gcp/dtensor-app-naive.py --prefix=gs://${GCS_BUCKET}"

  bash cluster-run.sh "conda activate py310; python dtensor-gpu-gcp/dtensor-keras-bert.py --prefix=gs://${GCS_BUCKET}"

When done, delete the cluster with,

  bash cluster-delete.sh
EOF

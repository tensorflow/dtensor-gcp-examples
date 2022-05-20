#! bin/bash
#
# Configures a GPU VM node for DTensor.

# This command produces cluster-run.sh, which can be used to run command
# on the node.
#
# The git repo is cloned to the VMs.

export IMAGE_FAMILY=common-cu113
export ZONE=us-west1-b
export INSTANCE_TYPE="n1-standard-8"
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
bash cluster-run.sh "conda activate py310; pip install -q tf-nightly"
bash cluster-bcast.sh launch ./
bash cluster-run.sh "if ! [[ -d dtensor-gpu-gcp ]]; then git clone https://github.com/rainwoodman/dtensor-gpu-gcp; fi"
bash cluster-run.sh "cd dtensor-gpu-gcp; git pull"
bash cluster-run.sh "ls -l dtensor-gpu-gcp;"

cat <<EOF
Next, run the application with 4 clients:

  bash cluster-run.sh "conda activate py310; ./launch python dtensor-gpu-gcp/dtensor-app-naive.py"

As there only 1 node, you can also run the application with a single client.

  bash cluster-run.sh "conda activate py310; python dtensor-gpu-gcp/dtensor-app-naive.py"

When done, delete the cluster with,

  bash cluster-delete.sh
EOF

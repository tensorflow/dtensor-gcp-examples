#! bin/bash
#
# Configures a GPU VM node for DTensor.

# This command produces cluster-run.sh, which can be used to run command
# on the node.
#
# The git repo is cloned to the VMs.

export IMAGE_FAMILY=tf-ent-2-9-cu113
export ZONE=us-west1-b
export INSTANCE_TYPE="n1-standard-8"
export NAME="dtensor-singlenode"
export PORT=9898
export NUM_GPUS=4

INSTANCES=($NAME)

bash `dirname $0`/../make-cluster-commands.sh "${ZONE}" "${INSTANCES[@]}"

gcloud compute instances create $NAME \
     --zone=$ZONE    \
     --image-family=$IMAGE_FAMILY     \
     --image-project=deeplearning-platform-release   \
     --maintenance-policy=TERMINATE   \
     --accelerator="type=nvidia-tesla-t4,count=${NUM_GPUS}"    \
     --machine-type=$INSTANCE_TYPE     \
     --boot-disk-size=120GB   \
     --metadata="install-nvidia-driver=True"  \

while bash cluster-run.sh ls |grep 'exited with return code'; do
  echo Health checking
  sleep 10
done

bash cluster-bcast.sh launch ./
bash cluster-run.sh "if ! [[ -d dtensor-gpu-gcp ]]; then git clone https://github.com/rainwoodman/dtensor-gpu-gcp; fi"
bash cluster-run.sh "cd dtensor-gpu-gcp; git pull"
bash cluster-run.sh "ls -l dtensor-gpu-gcp;"

cat <<EOF
Next, run the application with 4 clients:

  bash cluster-run.sh "./launch python dtensor-gpu-gcp/dtensor-app-naive.py"

As there only 1 node, you can also run the application with a single client.

  bash cluster-run.sh "python dtensor-gpu-gcp/dtensor-app-naive.py"

When done, delete the cluster with,

  bash cluster-delete.sh
EOF

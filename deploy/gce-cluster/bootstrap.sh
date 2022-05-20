#! bin/bash
#
# Configures a GPU VM cluster for DTensor.

# This command produces cluster-run.sh, which can be used to run command
# on all nodes of the cluster.
#
# The git repo is cloned to the VMs.

export IMAGE_FAMILY=common-cu113
export ZONE=us-west1-b
export INSTANCE_TYPE="n1-standard-8"
export COUNT=4
export NAME_PREFIX="dtensor-cluster"
export PORT=9898
export TAGS="dtensor-cluster-node"


INSTANCES=()
for i in `seq -s " " -f %03g 1 $COUNT`; do
  INSTANCES+=("${NAME_PREFIX}-${i}")
done

bash `dirname $0`/../make-cluster-commands.sh "${ZONE}" "${INSTANCES[@]}"

gcloud compute instances bulk create --predefined-names=$(printf "%s," ${INSTANCES[@]} | sed 's/,$//') \
     --zone=$ZONE    \
     --image-family=$IMAGE_FAMILY     \
     --image-project=deeplearning-platform-release   \
     --on-host-maintenance=TERMINATE   \
     --accelerator="type=nvidia-tesla-t4,count=2"    \
     --machine-type=$INSTANCE_TYPE     \
     --boot-disk-size=120GB   \
     --scopes=default,storage-rw \
     --metadata="install-nvidia-driver=True"  \
     --count=4 \
     --tags=${TAGS}

while bash cluster-run.sh ls |grep 'exited with return code'; do
  echo Health checking
  sleep 10
done

for i in ${INSTANCES[@]}; do
  echo "${i}:${PORT}"
done > dtensor-jobs

bash cluster-run.sh "sudo /opt/conda/bin/conda clean -q -y --all"
bash cluster-run.sh "conda create -q -y -n py310 python=3.10"
bash cluster-run.sh "conda activate py310; pip install -q tf-nightly"

bash cluster-bcast.sh dtensor-jobs ./
bash cluster-bcast.sh launch ./

bash cluster-run.sh "if ! [[ -d dtensor-gpu-gcp ]]; then git clone https://github.com/rainwoodman/dtensor-gpu-gcp; fi"
bash cluster-run.sh "cd dtensor-gpu-gcp; git pull"
bash cluster-run.sh "ls -l dtensor-gpu-gcp;"

cat <<EOF
Next, run the application with,

  bash cluster-run.sh "conda activate py310; ./launch python dtensor-gpu-gcp/dtensor-app-naive.py"

When done, delete the cluster with,

  bash cluster-delete.sh
EOF

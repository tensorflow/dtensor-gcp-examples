#! bin/bash
#
# Configures a GPU VM node for DTensor.

# This command produces single-node-run.sh, which can be used to run command
# on the node.
#
# The git repo is cloned to the VMs.

export IMAGE_FAMILY=tf-ent-2-9-cu113
export ZONE=us-west1-b
export INSTANCE_TYPE="n1-standard-8"
export NAME="dtensor-singlenode"
export PORT=9898
export NUM_GPUS=4
# If there is a proxy, use it (For Googlers).
if which corp-ssh-helper > /dev/null; then
  EXTRA_SSH_ARGS="-o ProxyCommand='corp-ssh-helper %h %p'"
else
  EXTRA_SSH_ARGS=
fi

INSTANCES=($NAME)

cat > single-node-run.sh <<EOF
for i in ${INSTANCES[@]}; do
  echo Running on "\${i}" "\$*"
  gcloud compute ssh \$i --zone=$ZONE -- -q -n ${EXTRA_SSH_ARGS} bash -c -l "'\$*'" > /tmp/\${i}.log 2>&1 &
done
wait
for i in ${INSTANCES[@]}; do
  echo ===Log from "\${i}"===
  cat /tmp/\${i}.log
done
EOF

cat > single-node-delete.sh <<EOF
gcloud compute instances delete ${INSTANCES[@]} --zone=$ZONE 
EOF

gcloud compute instances create $NAME \
     --zone=$ZONE    \
     --image-family=$IMAGE_FAMILY     \
     --image-project=deeplearning-platform-release   \
     --maintenance-policy=TERMINATE   \
     --accelerator="type=nvidia-tesla-t4,count=${NUM_GPUS}"    \
     --machine-type=$INSTANCE_TYPE     \
     --boot-disk-size=120GB   \
     --metadata="install-nvidia-driver=True"  \

while bash single-node-run.sh ls |grep 'exited with return code'; do
  echo Health checking
  sleep 10
done

bash single-node-run.sh "if ! [[ -d dtensor-gpu-gcp ]]; then git clone https://github.com/rainwoodman/dtensor-gpu-gcp; fi"
bash single-node-run.sh "cd dtensor-gpu-gcp; git pull"
bash single-node-run.sh "ls -l dtensor-gpu-gcp;"

echo "Next, run the clients with,"
echo '  bash single-node-run.sh "cd dtensor-gpu-gcp; bash run-single-node.sh"'
echo "When done, delete the single-node with,"
echo '  bash single-node-delete.sh '

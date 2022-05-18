#! bin/bash
#
# Configures a GPU VM cluster for DTensor.

# This command produces cluster-run.sh, which can be used to run command
# on all nodes of the cluster.
#
# The git repo is cloned to the VMs.

export IMAGE_FAMILY=tf-ent-2-9-cu113
export ZONE=us-west1-b
export INSTANCE_TYPE="n1-standard-8"
export COUNT=4
export NAME_PREFIX="dtensor-cluster"
export PORT=9898

# If there is a proxy, use it (For Googlers).
if which corp-ssh-helper > /dev/null; then
  EXTRA_SSH_ARGS="-o ProxyCommand='corp-ssh-helper %h %p'"
else
  EXTRA_SSH_ARGS=
fi

INSTANCES=()
for i in `seq -s " " -f %03g 1 $COUNT`; do
  INSTANCES+=("${NAME_PREFIX}-${i}")
done

cat > /tmp/bootstrap-dtensor-jobs.sh <<EOF
for i in ${INSTANCES[@]}; do
  echo "\${i}:${PORT}"
done > /etc/dtensor-jobs
EOF

cat > cluster-run.sh <<EOF
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

cat > cluster-delete.sh <<EOF
gcloud compute instances delete ${INSTANCES[@]} --zone=$ZONE 
EOF

gcloud compute instances bulk create --name-pattern="${NAME_PREFIX}-###" \
     --zone=$ZONE    \
     --image-family=$IMAGE_FAMILY     \
     --image-project=deeplearning-platform-release   \
     --on-host-maintenance=TERMINATE   \
     --accelerator="type=nvidia-tesla-t4,count=1"    \
     --machine-type=$INSTANCE_TYPE     \
     --boot-disk-size=120GB   \
     --metadata="install-nvidia-driver=True"  \
     --count=4 \
     --metadata-from-file=startup-script=/tmp/bootstrap-dtensor-jobs.sh

while bash cluster-run.sh ls /etc/dtensor-jobs |grep 'No such file\|exited with return code'; do
  echo Health checking
  sleep 10
done

bash cluster-run.sh cat /etc/dtensor-jobs

bash cluster-run.sh "if ! [[ -d dtensor-gpu-gcp ]]; then git clone https://github.com/rainwoodman/dtensor-gpu-gcp; fi"
bash cluster-run.sh "cd dtensor-gpu-gcp; git pull"
bash cluster-run.sh "ls -l dtensor-gpu-gcp;"

echo "Next, run the clients with,"
echo '  bash cluster-run.sh "cd dtensor-gpu-gcp; bash run-multi-nodes.sh"'

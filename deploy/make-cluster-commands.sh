#! /bin/bash

ZONE=$1
shift
INSTANCES=($*)

# If there is a proxy, use it (For Googlers).
if which corp-ssh-helper > /dev/null; then
  EXTRA_SSH_ARGS="-o ProxyCommand=corp-ssh-helper %h %p"
else
  EXTRA_SSH_ARGS=
fi

cat > cluster-run.sh <<EOF
PIDS=()
mkdir -p /tmp/dtensor/pids
for i in ${INSTANCES[@]}; do
  echo Running on "\${i}" "\$*"
  gcloud compute ssh \$i --zone=us-west1-b -- -t -q -n "-o ProxyCommand=corp-ssh-helper %h %p" bash -c -l "'\$*'" > /tmp/\${i}.log 2>&1 &
  CPID=\$!
  PIDS+=("\${CPID}")
  echo \$i > "/tmp/dtensor/pids/\${CPID}"
done

while [[ -n "\${PIDS}" ]]; do
  wait -p CPID -fn "\${PIDS[@]}"
  PIDS=(\${PIDS[@]/\$CPID})
  NODE=\$(cat /tmp/dtensor/pids/\${CPID})
  echo ===Log from "\${NODE}"===
  cat /tmp/\${NODE}.log
done
EOF

cat > cluster-bcast.sh <<EOF
echo Broadcasting \$1 to ${INSTANCES[@]}
for i in ${INSTANCES[@]}; do
  ( gcloud compute scp --zone=$ZONE --scp-flag "${EXTRA_SSH_ARGS}" \$1 \$i:\$2 )&
done
wait
EOF

cat > cluster-delete.sh <<EOF
gcloud compute instances delete ${INSTANCES[@]} --zone=$ZONE
EOF


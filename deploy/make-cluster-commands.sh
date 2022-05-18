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
for i in ${INSTANCES[@]}; do
  echo Running on "\${i}" "\$*"
  gcloud compute ssh \$i --zone=$ZONE -- -t -q -n "${EXTRA_SSH_ARGS}" bash -c -l "'\$*'" > /tmp/\${i}.log 2>&1 &
done
wait
for i in ${INSTANCES[@]}; do
  echo ===Log from "\${i}"===
  cat /tmp/\${i}.log
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


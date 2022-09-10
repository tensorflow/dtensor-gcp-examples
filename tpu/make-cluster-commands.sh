#! /bin/bash
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


NAME=$1
ZONE=$2
NUM_WORKERS=$3

# If there is a proxy, use it (For Googlers).
if which corp-ssh-helper > /dev/null; then
  EXTRA_SSH_ARGS="-o ProxyCommand=corp-ssh-helper %h %p"
else
  EXTRA_SSH_ARGS=
fi

cat > cluster-run.sh <<EOF
trap 'trap "" SIGTERM; kill 0; wait; ' EXIT

PIDS=()
mkdir -p /tmp/dtensor/pids
for ((i=0;i<$NUM_WORKERS;i++)); do
  echo Running on worker "\${i}" "\$*"
  gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=\$i -- -t -q -n "-o ProxyCommand=corp-ssh-helper %h %p" bash -c -l "'\$*'" > /tmp/${NAME}_\${i}.log 2>&1 &
  CPID=\$!
  PIDS+=("\${CPID}")
  echo \$i > "/tmp/dtensor/pids/\${CPID}"
done

for CPID in \${PIDS[@]}; do
  NODE=\$(cat /tmp/dtensor/pids/\${CPID})
  echo ===Log from "\${NODE}"===
  tail -c +0 -f /tmp/${NAME}_\${NODE}.log &
  TPID=\$!
  wait -fn \${CPID}
  kill \${TPID}
done
EOF

cat > cluster-bcast.sh <<EOF
echo Broadcasting \$1 to all workers of $NAME
gcloud compute tpus tpu-vm scp --zone=$ZONE --worker=all --scp-flag "${EXTRA_SSH_ARGS}" \$1 ${NAME}:\$2 &
wait
EOF

cat > cluster-delete.sh <<EOF
gcloud compute tpus tpu-vm delete $NAME --zone=$ZONE
EOF


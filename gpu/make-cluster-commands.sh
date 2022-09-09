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
shift; shift
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
  gcloud compute ssh \$i --zone=$ZONE -- -t -q -n "-o ProxyCommand=corp-ssh-helper %h %p" bash -c -l "'\$*'" > /tmp/\${i}.log 2>&1 &
  CPID=\$!
  PIDS+=("\${CPID}")
  echo \$i > "/tmp/dtensor/pids/\${CPID}"
done

for CPID in \${PIDS[@]}; do
  wait -fn \${CPID}
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


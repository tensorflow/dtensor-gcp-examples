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


DTENSOR_JOBS_FILE=`dirname $0`/dtensor-jobs

if ! [[ -f "${DTENSOR_JOBS_FILE}" ]]; then
  echo "${DTENSOR_JOBS_FILE}" is missing. Ensure the cluster is bootstraped.
fi
content=$(printf "%s," $(cat ${DTENSOR_JOBS_FILE}))
export DTENSOR_JOBS="${content%,}"

HOSTNAME=$(hostname)
export TF_CPP_MIN_LOG_LEVEL=3
export DTENSOR_JOB_NAME=worker
export DTENSOR_NUM_CLIENTS=`python get-clients.py`

for i in $(python get-clients.py ${HOSTNAME}); do
  export DTENSOR_CLIENT_ID="$i"
  ( exec $* ) &
done
wait

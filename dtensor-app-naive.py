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
"""A simple dtensor application.

The following script is a simple application using the DTensor API.

It performs a global reduce sum on a mesh of N devices, then performs 
checkpoint save and restore.
The N devices can be on the same client or on different clients.

Refer to the project README for how to use this script on GCP.

Sample usage:

1. Run on a single host with 8 GPUs:

```
python dtensor-app-naive.py --device_type=GPU --num_global_devices=8
```

2. Run on 2 hosts with 16 GPUs:
(The port number 9991 is an arbitrary free port on the hosts.)

```For host1
env DTENSOR_CLIENT_ID=0 DTENSOR_NUM_CLIENTS=2 \
    DTENSOR_JOB_NAME=training \
    DTENSOR_JOBS=host1:9991,host2:9991 \
python dtensor-app-naive.py --device_type=GPU --num_global_devices=16
```

```For host2
env DTENSOR_CLIENT_ID=1 DTENSOR_NUM_CLIENTS=2 \
    DTENSOR_JOB_NAME=training \
    DTENSOR_JOBS=host1:9991,host2:9991 \
python dtensor-app-naive.py --device_type=GPU --num_global_devices=16
```

"""
import argparse
import os

import tensorflow as tf
from tensorflow.experimental import dtensor

ap = argparse.ArgumentParser()
ap.add_argument(
    '--ckpt_path_prefix',
    default='gs://dtensor-checkpoints/app-naive',
    help='prefix for checkpointing')
ap.add_argument(
    '--device_type',
    default='GPU',
    choices=['GPU', 'TPU', 'CPU'],
    help='device type')
ap.add_argument(
    '--num_global_devices',
    type=int,
    default=8,
    help='Expected number of global accelerator devices for the run. '
         'If different from number of available devices an error is raised.')


# Patch _DVariableSaveable for GPU loading. See b/236027284 for more details.
from tensorflow.dtensor.python.d_variable import _DVariableSaveable

def restore(self, restored_tensors, restored_shapes):
  """Restores the same value into all variables."""
  tensor, = restored_tensors

  if self._original_layout.mesh.device_type().upper() != 'CPU':
    with tf.device(self._dvariable.device):
      tensor = dtensor.pack(
          dtensor.unpack(tensor), self._original_layout)
  return self._dvariable.assign(
      tf.cast(tensor, dtype=self._dvariable.dtype) if self._dvariable
      .save_as_bf16 else tensor)


_DVariableSaveable.restore = restore


def configure_virtual_devices(num_device, device_type):
  phy_devices = tf.config.list_physical_devices(device_type)
  device_config = tf.config.LogicalDeviceConfiguration()
  if len(phy_devices) >= num_device:
    for n in range(num_device):
      tf.config.set_logical_device_configuration(phy_devices[n],
                                                 [device_config])
  else:
    phy_to_logical_ratio = num_device // len(phy_devices)
    for n in range(len(phy_devices)):
      print(f'Config for device id {n}')
      tf.config.set_logical_device_configuration(phy_devices[n], [
          device_config,
      ] * phy_to_logical_ratio)
  return [f'{device_type}:{i}' for i in range(num_device)]


# ============================ Main =======================================


def main():
  args = ap.parse_args()

  print(tf.__version__)

  # CPU device is needed for checkpoint, even when running on GPU mesh
  # we need to config 1:1 mapping between accelerator and CPU for checkpoint.
  # This need to happen before we init dtensor multi client.
  configure_virtual_devices(args.num_global_devices // dtensor.num_clients(), 'CPU')

  if args.device_type == 'GPU':
    dtensor.initialize_multi_client()
  elif args.device_type == 'TPU':
    tf.experimental.dtensor.initialize_tpu_system()
  else:  # args.device_type == 'CPU'
    dtensor.initialize_multi_client()

  print(f'Using {dtensor.num_local_devices(args.device_type)} local devices '
        f'of type {args.device_type}, with {dtensor.num_clients()} clients.')

  if dtensor.num_global_devices(args.device_type) != args.num_global_devices:
    raise ValueError(f'Expect {args.num_accelerator} physical devices for '
                   f'{args.device_type}, got device list: {gpu_devices}')

  mesh = dtensor.create_distributed_mesh(
      [('batch', dtensor.num_global_devices(args.device_type))],
      device_type=args.device_type)

  layout = dtensor.Layout(['batch', dtensor.UNSHARDED], mesh)
  data = dtensor.call_with_layout(tf.ones, layout, shape=(32, 100))
  print(data)

  # Cross-client reduction
  print(tf.reduce_sum(data))

  # Checkpointing
  v = dtensor.DVariable(data)
  cpt = dtensor.DTensorCheckpoint(mesh=mesh, v=v)
  saved_path = cpt.save(os.path.join(args.ckpt_path_prefix, 'checkpoint-1'))
  # Restoring checkpoint
  cpt.restore(saved_path)


main()

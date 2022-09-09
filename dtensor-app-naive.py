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
"""This is a naive dtensor application.

It performs a global reduce sum on a mesh of N devices. The N devices can
be on the same client or on different clients.
"""
import argparse
import os

import tensorflow as tf
from tensorflow.experimental import dtensor

ap = argparse.ArgumentParser()
ap.add_argument(
    '--prefix',
    default='gs://dtensor-checkpoints',
    help='prefix for checkpointing')
ap.add_argument(
    '--device-type',
    default='GPU',
    choices=['GPU', 'TPU', 'CPU'],
    help='device type')
ap.add_argument(
    '--num-global-devices',
    type=int,
    default=8,
    help='number of global devices (only applies to non-TPU devices)')

def configure_virtual_cpus(ncpu):
  phy_devices = tf.config.list_physical_devices('CPU')
  tf.config.set_logical_device_configuration(phy_devices[0], [
      tf.config.LogicalDeviceConfiguration(),
  ] * ncpu)


def main():
  args = ap.parse_args()

  print(tf.__version__)

  if args.device_type != 'TPU':
    # Checkpointing requires the same number (or more) of logical CPU devices
    # as the number of GPU devices.
    num_global_devices = args.num_global_devices
    configure_virtual_cpus(num_global_devices // dtensor.num_clients())
    # Initializes multi-client DTensor.
    dtensor.initialize_multi_client()
  else:
    num_global_devices = None  # all TPU devices will be used.
    # Initialize the TPU system. Also initializes multi-client DTensor.
    dtensor.initialize_tpu_system()

  print("Device type:", args.device_type)
  print("Num local devices:", dtensor.num_local_devices(args.device_type))
  print("Num global devices:", dtensor.num_global_devices(args.device_type))

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
  saved_path = cpt.save(os.path.join(args.prefix, 'checkpoint-1'))
  # Restoring checkpoint
  cpt.restore(saved_path)


# Patch the _DVariableSaveable for GPU loading. See b/236027284 for more details.
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

main()

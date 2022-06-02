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

It performs a global reduce sum on a mesh of 4 devices. The 4 devices can
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

print('client', dtensor.client_id(), 'devices',
      tf.config.list_physical_devices('GPU'))
print(tf.__version__)


def configure_virtual_cpus(ncpu):
  phy_devices = tf.config.list_physical_devices('CPU')
  tf.config.set_logical_device_configuration(phy_devices[0], [
      tf.config.LogicalDeviceConfiguration(),
  ] * ncpu)


# Checkpointing requires the same number (or more) of logical CPU devices
# as the number of GPU devices.
configure_virtual_cpus(8)

dtensor.initialize_multi_client()

mesh = dtensor.create_distributed_mesh([('batch', 8)],
                                       device_type='GPU',
                                       num_global_devices=8)


def main():
  args = ap.parse_args()

  layout = dtensor.Layout(['batch', dtensor.UNSHARDED], mesh)
  data = dtensor.call_with_layout(tf.ones, layout, shape=(16, 100))
  print(data)

  # Cross-client reduction
  print(tf.reduce_sum(data))

  # Checkpointing
  v = dtensor.DVariable(data)
  cpt = dtensor.DTensorCheckpoint(mesh=mesh, v=v)
  cpt.save(os.path.join(args.prefix, 'checkpoint-1'))


main()

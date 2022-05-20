"""This is a naive dtensor application.

It performs a global reduce sum on a mesh of 4 devices. The 4 devices can
be on the same client or on different clients.
"""

import tensorflow as tf
from tensorflow.experimental import dtensor

print('client', dtensor.client_id(), 'devices', tf.config.list_physical_devices('GPU'))
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

mesh = dtensor.create_distributed_mesh([("batch", 8)], device_type='GPU', num_global_devices=8)

def main():

  layout = dtensor.Layout(['batch', dtensor.UNSHARDED], mesh)
  data = dtensor.call_with_layout(tf.ones, layout, shape=(16, 100))
  print(data)
  print(tf.reduce_sum(data))

  # Checkpointing
  v = dtensor.DVariable(data)
  cpt = dtensor.DTensorCheckpoint(mesh=mesh, v=v)
  cpt.save('gs://dtensor-checkpoints/checkpoint-1')

main()

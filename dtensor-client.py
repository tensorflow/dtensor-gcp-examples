import tensorflow as tf
from tensorflow.experimental import dtensor
print('client', dtensor.client_id(), 'devices', tf.config.list_physical_devices('GPU'))
print(tf.__version__)
dtensor.initialize_multi_client()
mesh = dtensor.create_distributed_mesh([("batch", 4)], device_type='GPU', num_global_devices=4)

def main():
    layout = dtensor.Layout(['batch', dtensor.UNSHARDED], mesh)
    data = dtensor.call_with_layout(tf.ones, layout, shape=(4, 100))
    print(data)
    print(tf.reduce_sum(data))

main()

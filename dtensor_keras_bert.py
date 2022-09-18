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
r"""Pretrain a bert model with tf.dtensor API.

The following script is a self-contained Bert model pretrain workflow, which
include model building, DTensor mesh setup and training loop.

This script requires tf-nightly and tf-models-nightly as dependency.

To train the model, it requires access to a preprocessed data directory. We have
the default GCS bucket that contains the data, but user might choose to use
their own data. The instructions for how to preprocess the data can be found in
https://github.com/keras-team/keras-nlp/tree/master/examples/bert#downloading-pretraining-data

Sample usage:

1. Train a "tiny" Bert model with model+data parallel on single host with 8 GPU.
This will run the Bert model with a 4x2 mesh (data x model).

```
python dtensor_bert_train.py --model_size=tiny --device_type=GPU \
  --num_accelerator=8 --distribution_mode=batchmodel
```

2. Train a "base" Bert model with data parallel on single host with 8 GPU.
This will run the Bert model with data parallel only.

```
python dtensor_bert_train.py --model_size=base --device_type=GPU \
  --num_global_devices=8 --distribution_mode=batch
```

3. Train a "base" Bert model with model+data parallel on single host with 8 GPU.
This will run the Bert model with a 2x4 mesh (data x model), which is different
from the usage1.

```
python dtensor_bert_train.py --model_size=base --device_type=GPU \
  --num_global_devices=8 --distribution_mode=batchmodel --model_parallel_dim_size=4
```

4. Train a "base" Bert model with model+data parallel on a V2-8 TPU.
This will run the Bert model with a 4x2 mesh (data x model).
! Note that the TPU instance need run with in as "TPU VM architecture" (1 VM).

```
python dtensor_bert_train.py --model_size=base --device_type=TPU \
  --num_global_devices=8 --distribution_mode=batchmodel
```

5. Train a "base" Bert model with model+data parallel on a multi-client setting
(2 hosts with 8 GPUs on each). please change the host1_address, host2_address
and port in the following commmand accordingly. The following config will run
a distributed mesh with 8x2 (data x model).

```For host1
env DTENSOR_CLIENT_ID=0 DTENSOR_NUM_CLIENTS=2 \
    DTENSOR_JOB_NAME=training \
    DTENSOR_JOBS=host1:9991,host2:9991 \
python dtensor_bert_train.py --model_size=base --device_type=GPU \
  --num_global_devices=16
```

```For host2
env DTENSOR_CLIENT_ID=1 DTENSOR_NUM_CLIENTS=2 \
    DTENSOR_JOB_NAME=training \
    DTENSOR_JOBS=host1:9991,host2:9991 \
python dtensor_bert_train.py --model_size=base --device_type=GPU \
  --num_global_devices=16
```

"""

import argparse
import os
import time

import tensorflow as tf

from tensorflow.experimental import dtensor
from tensorflow_models import nlp

ap = argparse.ArgumentParser()
ap.add_argument(
    '--ckpt_path_prefix',
    default='gs://scottzhu-dtensor-test/bert-small-checkpoint',
    help='prefix for checkpointing, can be a gs:// path or a local directory')
ap.add_argument(
    '--data_path',
    default='gs://chenmoney-testing/bert-pretraining-data-512-76/bert-pretraining-data/shard_*.tfrecord',
    help='file path for a training data. Can be a gs:// path or a local directory'
)
ap.add_argument(
    '--tensorboard_path',
    default='/tmp/dtensor_test/',
    help='The root directory that will be used for tensorboard logging. Sub '
    'direcotry will be created based on the timestep as well as model setting '
    'to group the related runs together.')
ap.add_argument(
    '--model_size',
    default='small',
    choices=['small', 'tiny', 'medium', 'base'],
    help='BERT model size setting.')
ap.add_argument(
    '--device_type',
    default='GPU',
    choices=['TPU', 'GPU', 'CPU'],
    help='device type')
ap.add_argument(
    '--num_global_devices',
    default=8,
    type=int,
    help='Expected number of global accelerator devices for the run. '
         'If different from number of available devices an error is raised.')
ap.add_argument(
    '--distribution_mode',
    default='batchmodel',
    choices=['batch', 'model', 'batchmodel'],
    help='distribution setting for the model and inputs')
ap.add_argument(
    '--enable_profile_trace',
    default=False,
    type=bool,
    help='Whether to run tensorboard profile tracing for performance debug.')
ap.add_argument(
    '--model_parallel_dim_size',
    default=2,
    type=int,
    help='model parallel dim size')
ap.add_argument(
    '--truncate_sequence_length',
    default=0,
    type=int,
    help='Truncates sequence in pretraining and data. 0 means no truncation.'
         'Setting this to a small number (32) drastically increases training '
         'speed.')

# Parameters for distribution(dtensor)

BATCH_DIM = 'x'
MODEL_DIM = 'y'


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


# Hparams for a bert model
vocab_size = 30522
num_masked_tokens = 76
data_sequence_length = 512

# Training config
batch_size = 128
training_step = int(256 / batch_size) * 500 * 1000


def get_model_setting(model_size):
  if model_size == 'tiny':
    return {
        'num_layers': 2,
        'hidden_size': 128,
        'num_attention_heads': 2,
        'inner_size': 512,
        'num_classes': 2,
    }
  elif model_size == 'small':
    return {
        'num_layers': 4,
        'hidden_size': 512,
        'num_attention_heads': 8,
        'inner_size': 2048,
        'num_classes': 2,
    }
  elif model_size == 'medium':
    return {
        'num_layers': 8,
        'hidden_size': 512,
        'num_attention_heads': 8,
        'inner_size': 2048,
        'num_classes': 2,
    }
  elif model_size == 'base':
    return {
        'num_layers': 12,
        'hidden_size': 768,
        'num_attention_heads': 12,
        'inner_size': 3072,
        'num_classes': 2,
    }
  else:
    raise ValueError(f'Invalid model size setting {model_size}')


# ==================================== Data =================================


def decode_record(record):
  """Decodes a record to a TensorFlow example."""
  seq_length = data_sequence_length  # Should it be 512?
  lm_length = num_masked_tokens  # Should this be 76?
  name_to_features = {
      'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
      'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
      'masked_lm_positions': tf.io.FixedLenFeature([lm_length], tf.int64),
      'masked_lm_ids': tf.io.FixedLenFeature([lm_length], tf.int64),
      'masked_lm_weights': tf.io.FixedLenFeature([lm_length], tf.float32),
      'next_sentence_labels': tf.io.FixedLenFeature([1], tf.int64),
  }
  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32 if needed.
  example = tf.io.parse_single_example(record, name_to_features)
  # for name in list(example.keys()):
  #     value = example[name]
  #     if value.dtype == tf.int64:
  #         value = tf.cast(value, tf.int32)
  #     example[name] = value
  return example


# Inspect the tf record and decode it
def preview_tfrecord(filepath):
  """Pretty prints a single record from a tfrecord file."""
  dataset = tf.data.TFRecordDataset(os.path.expanduser(filepath))
  num_total = 0
  num_dense = 0
  early_end = 5000000000
  for item in dataset:
    if num_total >= early_end:
      break
    num_total += 1

    example = tf.train.Example()
    example.ParseFromString(item.numpy())

    decoded = decode_record(item)
    input_mask = decoded['input_mask']
    num_valid = tf.reduce_sum(input_mask)
    if num_valid > 480:
      num_dense += 1
  print(f'TOTAL NUMBER OF RECORDS: {num_total}')
  print(f'NUMBER OF DENSE RECORDS: {num_dense}')


# This is only needed for matchinng the keras NLP data to model garden bert model
def remap_input_keys(data_entry):
  data_entry['input_word_ids'] = data_entry.pop('input_ids')
  data_entry['input_type_ids'] = data_entry.pop('segment_ids')
  return data_entry


# Create dataset
def create_dataset(data_file_path, batch_size, max_sequence_length):

  data_files = tf.io.gfile.glob(data_file_path)

  dataset = tf.data.TFRecordDataset(data_files, num_parallel_reads=10)
  dataset = dataset.map(
      decode_record,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )
  if max_sequence_length != data_sequence_length:

    def truncate_data(data):
      data['input_ids'] = data['input_ids'][:max_sequence_length]
      data['input_mask'] = data['input_mask'][:max_sequence_length]
      data['segment_ids'] = data['segment_ids'][:max_sequence_length]
      return data

    dataset = dataset.map(
        truncate_data,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
  dataset = dataset.map(
      remap_input_keys,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.shuffle(100)
  dataset = dataset.repeat().prefetch(buffer_size=10)
  return dataset


def package_inputs_to_dtensor(data, mesh):
  results = {}
  for key, inputs in data.items():
    target_layout = dtensor.Layout.batch_sharded(
        mesh, batch_dim=BATCH_DIM, rank=len(inputs.shape))
    replicated_layout = dtensor.Layout.replicated(
        mesh, rank=len(inputs.shape))
    d_input = dtensor.copy_to_mesh(inputs, replicated_layout)
    d_input = dtensor.relayout(d_input, target_layout)
    results[key] = d_input

  return results


# ==================================== Model =================================
# This is referred from google3/third_party/tensorflow_models/google/dtensor_models/sharding.py
def create_model_parallel_layout_map(mesh):
  layout_map = tf.keras.dtensor.experimental.LayoutMap(mesh=mesh)

  layout_map['.*word_embeddings.embeddings'] = dtensor.Layout(
      ['unsharded', 'y'], mesh)
  layout_map['.*pooler_transform.kernel'] = dtensor.Layout(
      ['unsharded', 'y'], mesh)
  layout_map['.*pooler_transform.bias'] = dtensor.Layout(['y'], mesh)
  layout_map['.*attention_layer.*key.*kernel'] = dtensor.Layout(
      ['unsharded', 'unsharded', 'y'], mesh)
  layout_map['.*attention_layer.*key.*bias'] = dtensor.Layout(
      ['y', 'unsharded'], mesh)
  layout_map[
      '.*attention_layer.*query.*kernel'] = dtensor.Layout(
          ['unsharded', 'unsharded', 'y'], mesh)
  layout_map['.*attention_layer.*query.*bias'] = dtensor.Layout(
      ['y', 'unsharded'], mesh)
  layout_map[
      '.*attention_layer.*value.*kernel'] = dtensor.Layout(
          ['unsharded', 'unsharded', 'y'], mesh)
  layout_map['.*attention_layer.*value.*bias'] = dtensor.Layout(
      ['y', 'unsharded'], mesh)
  layout_map[
      r'.*transformer/layer.\d*._output_dense.kernel'
  ] = dtensor.Layout(['y', 'unsharded'], mesh)
  layout_map[
      r'.*transformer/layer.\d*._output_dense.bias'
  ] = dtensor.Layout(['unsharded'], mesh)
  layout_map[r'predictions.transform.logits.kernel'] = dtensor.Layout(
      ['y', 'unsharded'], mesh)
  layout_map[r'cls/predictions.dense.kernel'] = dtensor.Layout(
      ['unsharded', 'y'], mesh)
  layout_map[r'cls/predictions.dense.bias'] = dtensor.Layout(
      ['y'], mesh)
  return layout_map


# Patch the MaskedLM layer in Bert model for dtensor performance.
def _gather_indexes(self, sequence_tensor: tf.Tensor, positions: tf.Tensor):
  """Gathers the vectors at the specific positions, for performance.

  Args:
      sequence_tensor: Sequence output of shape (`batch_size`, `seq_length`,
        num_hidden) where num_hidden is number of hidden units.
      positions: Positions ids of tokens in sequence to mask for pretraining
        of with dimension (batch_size, num_predictions) where
        `num_predictions` is maximum number of tokens to mask out and predict
        per each sequence.

  Returns:
      Masked out sequence tensor of shape (batch_size * num_predictions,
      num_hidden).
  """
  _, seq_length, width = sequence_tensor.shape.as_list()
  if seq_length is None:
    seq_length = tf.shape(sequence_tensor)[1]
  if width is None:
    width = tf.shape(sequence_tensor)[2]

  output_tensor = tf.gather(sequence_tensor, positions, batch_dims=1)
  return tf.reshape(output_tensor, [-1, width])


nlp.layers.MaskedLM._gather_indexes = _gather_indexes


def create_bert_model(mesh, model_size, max_sequence_length):
  layout_map = create_model_parallel_layout_map(mesh)

  model_setting = get_model_setting(model_size)
  num_layers = model_setting['num_layers']
  hidden_size = model_setting['hidden_size']
  num_attention_heads = model_setting['num_attention_heads']
  inner_size = model_setting['inner_size']
  num_classes = model_setting['num_classes']

  with tf.keras.dtensor.experimental.layout_map_scope(layout_map=layout_map):
    #!!! We need to fix this. The tf.gather doesn't support SPMD at the moment,
    # we have to force the use_one_hot code path to walkaround the issue.
    embedding = nlp.layers.OnDeviceEmbedding(
        vocab_size=vocab_size,
        embedding_width=hidden_size,
        initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        use_one_hot=True,
        name='word_embeddings')

    network = nlp.networks.BertEncoder(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        max_sequence_length=max_sequence_length,
        inner_dim=inner_size,
        embedding_layer=embedding)
    bert_pretrainer = nlp.models.BertPretrainer(
        network,
        num_classes=num_classes,
        num_token_predictions=num_masked_tokens,
        initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        activation='gelu')

  # for weight in bert_pretrainer.trainable_weights:
  #   print(f'{weight.name} has layout spec: {weight.layout.sharding_specs}')
  return bert_pretrainer


# ==================================== Training ================================


@tf.function
def train_step(bert_pretrainer, data, optimizer, metrics):
  with tf.GradientTape() as tape:
    output_dict = bert_pretrainer(data, training=True)
    lm_preds, nsp_preds = output_dict['masked_lm'], output_dict[
        'classification']
    lm_preds = tf.cast(lm_preds, tf.float32)
    nsp_preds = tf.cast(nsp_preds, tf.float32)
    lm_loss = tf.keras.losses.sparse_categorical_crossentropy(
        data['masked_lm_ids'], lm_preds, from_logits=True)
    lm_weights = data['masked_lm_weights']
    # lm_weights_summed = tf.reduce_sum(lm_weights, -1)
    # lm_loss = tf.reduce_sum(lm_loss * lm_weights, -1)
    lm_weights_summed = tf.reduce_sum(lm_weights)
    lm_loss = tf.reduce_sum(lm_loss * lm_weights)
    lm_loss = tf.math.divide_no_nan(lm_loss, lm_weights_summed)
    nsp_loss = tf.keras.losses.sparse_categorical_crossentropy(
        data['next_sentence_labels'], nsp_preds, from_logits=True)
    nsp_loss = tf.reduce_mean(nsp_loss)
    loss = lm_loss + nsp_loss

    # Compute gradients
  trainable_vars = bert_pretrainer.trainable_variables
  gradients = tape.gradient(loss, trainable_vars)
  # Update weights
  optimizer.apply_gradients(zip(gradients, trainable_vars))

  # Update metrics
  # Note that the lm_acc need the sample weights to filter out the padded
  # sequence.
  metrics['lm_accuracy'].update_state(data['masked_lm_ids'], lm_preds,
                                      data['masked_lm_weights'])
  metrics['nsp_accuracy'].update_state(data['next_sentence_labels'], nsp_preds)
  return {'loss': loss, 'lm_loss': lm_loss, 'nsp_loss': nsp_loss}


class LinearDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
  """
    A learning rate schedule with linear warmup and decay.
    This schedule implements a linear warmup for the first `num_warmup_steps`
    and a linear ramp down until `num_train_steps`.
  """

  def __init__(self, learning_rate, num_warmup_steps, num_train_steps):
    self.learning_rate = learning_rate
    self.warmup_steps = num_warmup_steps
    self.train_steps = num_train_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    peak_lr = tf.cast(self.learning_rate, dtype=tf.float32)
    warmup = tf.cast(self.warmup_steps, dtype=tf.float32)
    training = tf.cast(self.train_steps, dtype=tf.float32)

    # Linear Warmup will be implemented if current step is less than
    # `num_warmup_steps` else Linear Decay will be implemented.
    return tf.cond(
        step < warmup,
        lambda: peak_lr * (step / warmup),
        lambda: tf.math.maximum(
            0.0,
            peak_lr * (training - step) / (training - warmup)),
    )

  def get_config(self):
    return {
        'learning_rate': self.learning_rate,
        'num_warmup_steps': self.warmup_steps,
        'num_train_steps': self.train_steps,
    }


def create_optimizer(mesh):
  initial_learning_rate = 1e-4  # Original value from the MG is 1e-4, need to check why
  # learning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
  #     initial_learning_rate,
  #     decay_steps=training_step,
  #     end_learning_rate=0.0,
  #     cycle=False)
  learning_rate_schedule = LinearDecayWithWarmup(initial_learning_rate,
                                                 0.01 * training_step,
                                                 training_step)
  return tf.keras.dtensor.experimental.optimizers.Adam(
      learning_rate=learning_rate_schedule, mesh=mesh)


def create_metrics(mesh):
  lm_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='lm_accuracy', mesh=mesh)
  nsp_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
      name='nsp_accuracy', mesh=mesh)

  return {'lm_accuracy': lm_accuracy, 'nsp_accuracy': nsp_accuracy}


# ==================================== Checkpoint =================================

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


def config_checkpoint(checkpoint_dir):
  if not tf.io.gfile.exists(checkpoint_dir):
    tf.io.gfile.makedirs(checkpoint_dir)

  step_file_path = os.path.join(checkpoint_dir, 'steps')
  if tf.io.gfile.exists(step_file_path):
    with tf.io.gfile.GFile(step_file_path, 'r') as f:
      start_step = int(f.read())
      print('start up step: ', start_step)
  else:
    start_step = 0
    print('start up step: ', start_step)
    with tf.io.gfile.GFile(step_file_path, 'w') as f:
      f.write(str(start_step))
  return start_step


# ============================ Tensorboard =======================================
def config_tensorboard(logging_dir_path):
  import datetime
  current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  train_log_dir = os.path.join(logging_dir_path, current_time + 'train')
  tf.io.gfile.makedirs(train_log_dir)
  train_summary_writer = tf.summary.create_file_writer(train_log_dir)
  return train_summary_writer


def start_trace(logging_dir_path, mesh):
  dtensor.barrier(mesh)
  tf.summary.trace_on(graph=True, profiler=False)
  tf.profiler.experimental.start(logdir=logging_dir_path)


def end_trace(mesh):
  dtensor.barrier(mesh)
  tf.summary.trace_off()
  tf.profiler.experimental.stop(save=True)


# ============================ Main =======================================


def main():
  args = ap.parse_args()

  print('tensorflow version', tf.__version__)

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

  is_batch_parallel = 'batch' in args.distribution_mode
  is_model_parallel = 'model' in args.distribution_mode
  if is_batch_parallel and is_model_parallel:
    batch_parallel_dim = args.num_global_devices // args.model_parallel_dim_size
    model_parallel_dim = args.model_parallel_dim_size
  elif is_batch_parallel:
    batch_parallel_dim = args.num_global_devices
    model_parallel_dim = 1
  else:
    batch_parallel_dim = 1
    model_parallel_dim = args.num_global_devices

  mesh_dims = [
      (BATCH_DIM, batch_parallel_dim),
      (MODEL_DIM, model_parallel_dim),
  ]
  print(f'Mesh setting is: {mesh_dims}')

  mesh = dtensor.create_distributed_mesh(mesh_dims, device_type=args.device_type)

  # ============================ Training ============================
  max_sequence_length = args.truncate_sequence_length or data_sequence_length

  start_step = config_checkpoint(args.ckpt_path_prefix)

  tf.keras.utils.set_random_seed(1337)
  tf.keras.backend.experimental.enable_tf_random_generator()

  model = create_bert_model(
      mesh, model_size=args.model_size, max_sequence_length=max_sequence_length)
  metrics = create_metrics(mesh)
  optimizer = create_optimizer(mesh)
  optimizer.iterations.assign(start_step)
  dataset = create_dataset(
      args.data_path, batch_size, max_sequence_length=max_sequence_length)
  tensorboard_path = args.tensorboard_path
  tb_dir = os.path.join(
      tensorboard_path,
      f'bert_{args.model_size}_{args.distribution_mode}_{batch_parallel_dim}x{model_parallel_dim}',
      'tensorboard')
  train_summary_writer = config_tensorboard(tb_dir)
  enable_profile_trace = args.enable_profile_trace

  print(f'start_step is {start_step}')
  # Restore checkpoints
  if start_step != 0:
    cpt = dtensor.DTensorCheckpoint(mesh=mesh, root=model)

    # Find the checkpoint file based on the prefix
    # load_ckpt_path = tf.io.gfile.glob(checkpoint_path + "-*.index")
    load_ckpt_path = os.path.join(args.ckpt_path_prefix, 'ckpt-1')
    cpt.restore(load_ckpt_path)

  steps = start_step
  logging_steps = 100

  start_time = time.monotonic()
  for data in dataset:
    steps += 1

    # Trace the performance log for 1 step
    if steps % logging_steps == 0 and enable_profile_trace:
      start_trace(tb_dir, mesh)

    data = package_inputs_to_dtensor(data, mesh)
    losses = train_step(model, data, optimizer, metrics)

    if steps % logging_steps == 0 and enable_profile_trace:
      end_trace(mesh)

    if steps % logging_steps == 0:
      dtensor.barrier(mesh)
      end_time = time.monotonic()
      print('===========================')
      print(f'step: {steps}')
      print(f'Took: {end_time - start_time}')
      print(f'Steps per second: {logging_steps / (end_time - start_time)}')
      with train_summary_writer.as_default():
        for name, loss in losses.items():
          print(f'{name}: {loss.numpy()}')
          tf.summary.scalar(name, tf.math.reduce_mean(loss), step=steps)

        for name, metric in metrics.items():
          print(f'{name}: {metric.result().numpy()}')
          tf.summary.scalar(name, metric.result(), step=steps)
          metric.reset_state()
        print(f'current learning rate: {optimizer.lr.numpy()}')
        tf.summary.scalar('learning rate', optimizer.lr, step=steps)

      # Saving checkpoint
      cpt = dtensor.DTensorCheckpoint(mesh=mesh, root=model)
      cpt.save(os.path.join(args.ckpt_path_prefix, 'ckpt'))
      # Write down steps
      step_file_path = os.path.join(args.ckpt_path_prefix, 'steps')
      with tf.io.gfile.GFile(step_file_path, 'w') as f:
        f.write(str(steps))

      start_time = time.monotonic()

    if steps >= training_step:
      break


if __name__ == '__main__':
  main()

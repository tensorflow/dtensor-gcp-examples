# DTensor TPU Examples on GCP

This directory contains examples of using single-client or multi-client DTensor
on GCP TPU VMs.

## Description

- `bootstrap.sh <ACCELERATOR_TYPE>`: Starts the TPU cluster / node. The
  `ACCELERATOR_TYPE` value is specified as a TPU type and a topology. For
  example, `v2-8` specifies a `v2` TPU configuration with 8 TPU cores. `v3-2048`
  specifies a `v3` TPU Pod configuration with 2048 cores.

- `launch`: The application launcher for this cluster. It configures the DTensor
  environment variables before launching the command provided in the
  command-line as DTensor clients. This script (and dependency) are broadcast to
  the worker VMs and shall be run from the VMs (e.g. via cluster-run.sh).
  **This file is the most relevant to the deployemnt.**

- `cluster-run.sh`: (produced by bootstrap.sh) runs the provided command on all
  workers.

- `cluster-bcast.sh`: (produced by bootstrap.sh) copies the file to all workers.

- `cluster-delete.sh`: (produced by bootstrap.sh) deletes all workers in the
  cluster.


## Running as single-client

A single client TPU VM setup with 8 cores can be launched as below:

```
$ gcloud auth login ...

$ git clone ...
$ cd dtensor-gcp-examples/tpu

$ bash bootstrap.sh v2-8
$ bash cluster-run.sh "./launch python3 dtensor-gcp-examples/dtensor-app-naive.py --device_type=TPU"
$ bash cluster-delete.sh
```


## Running as multi-client

The same commands can be used to run a multi-client TPU VM setup. For example,
a 32 core cluster with 4 clients can be launched as below:

```
$ gcloud auth login ...

$ git clone ...
$ cd dtensor-gcp-examples/tpu

$ bash bootstrap.sh v2-32
$ bash cluster-run.sh "./launch python3 dtensor-gcp-examples/dtensor-app-naive.py --device_type=TPU"
$ bash cluster-delete.sh
```

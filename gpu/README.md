# DTensor GPU Examples on GCP

This directory contains examples of using single-client or multi-client DTensor
on GCP with a cluster of GPUs.


## Description

- `gce-cluster/`: First build a cluster of 4 GCE 2GPU VMs, then run 1
  DTensor client per VM. Note that the network configuration of these GPU VMs
  are likely not optimal.

- `gce-node/`: First build a single GCE GPU VM with 8 gpus, then run 1
  client per GPU on the VM. For each DTensor client, the unwanted GPUs are
  hidden via `CUDA_VISIBILE_GPU`.

In each of these directories:

- `bootstrap.sh`: Starts the cluster / node. Adding 'cpu' (running
  `bootstrap.sh cpu`) will start a CPU only cluster without GPUs.

- `launch`: The application launcher for this cluster. It configures the DTensor
  environment variables before launching the command provided in the
  command-line as DTensor clients. This script (and dependency) are broadcast to
  the VMs and shall be run from the VMs (e.g. via cluster-run.sh).
  **This file is the most relevant to the deployemnt.**

- `cluster-run.sh`: (produced by bootstrap.sh) runs the provided command on all
  VMs in the cluster.

- `cluster-bcast.sh`: (produced by bootstrap.sh) copies the file to all VMs.

- `cluster-delete.sh`: (produced by bootstrap.sh) deletes all VMs in the
  cluster.


## Running as multi-client

To run the application with multiple clients, e.g using the GCE cluster example:

```
$ gcloud auth login ...

$ git clone ...
$ cd dtensor-gcp-examples/gpu

# Run from the cluster deployment:
$ cd gce-cluster
$ bash bootstrap.sh
$ bash cluster-run.sh "conda activate py3; ./launch python dtensor-gcp-examples/dtensor_app_naive.py"
$ bash cluster-delete.sh
```

## Running as single-client

The gce-node deployment can also run with a single client.
Just skip `./launch` and run the script as a regular python application:

```
$ cd gce-node
$ bash bootstrap.sh
$ bash cluster-run.sh "conda activate py3; TF_CPP_MIN_LOG_LEVEL=3 python dtensor-gcp-examples/dtensor_app_naive.py"
$ bash cluster-delete.sh
```

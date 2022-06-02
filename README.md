# dtensor-gpu-gcp

This project contains an example of using multi-client DTensor on GCP with a
cluster of GPUs.

## Prerequirement

1. gcloud environment on the local console:
  ```
  gcloud auth login ...
  gcloud config set project  ...
  ```

2. A GCS bucket that the GCE service account can write into. The bucket is used
  to demo checkpointing. Set the prefix paths name with
  ```
  export GCS_BUCKET=<bucket_name>
  ```
  or edit bootstrap.sh.


## Description

- deploy/gce-cluster: First build a cluster of 4 GCE 2GPU VMs, then run 1 dtensor
  client per VM.

- deploy/gce-node: First build a single GCE GPU VM with 8 gpus, then run 1 client
  per GPU on the VM. For each dtensor client, the unwanted GPUs are hidden via
  `CUDA_VISIBILE_GPU`.

- dtensor-client.py: the dtensor application. The simple example creates a
  distributed tensor (a DTensor), and performs a collective reduction.
  The script only contains application logic, and is independent from the
  deployment site of choice.

In each `deploy/*` directory:

- bootstrap.sh: Starts the cluster / node. Adding 'cpu'
  (running bootstrap.sh cpu) will start a CPU only cluster without GPUs.

- launch: The application launcher for this cluster. It configures the
  DTensor environment variables before launching the command provided in
  the command-line as dtensor clients.
  This script (and dependency) are broadcast to the VMs and shall be run from
  the VMs (e.g. via cluster-run.sh).
  **This file is the most relevant to the deployemnt.**

- cluster-run.sh: (produced by bootstrap.sh) runs the provided command
  on all VMs in the cluster.

- cluster-bcast.sh: (produced by bootstrap.sh) copies the file to all VMs.

- cluster-delete.sh: (produced by bootstrap.sh) deletes all VMs
  in the cluster.

## Running as multi-client


To run the application with multiple clients, e.g using the GCE cluster example:

```
$ gcloud auth login ...

$ git clone ...
$ cd dtensor-gpu-gcp

# Run from the cluster deployment:
$ cd deploy/gce-cluster
$ bash bootstrap.sh
$ bash cluster-run.sh "conda activate py310; ./launch python dtensor-gpu-gcp/dtensor-app-naive.py"
$ bash cluster-delete.sh
```

## Running as single-client
The gce-node deployment can also run with a single client.
Just skip `./launch` and run the script as a regular python application:

```
$ cd deploy/gce-node
$ bash bootstrap.sh
$ bash cluster-run.sh "conda activate py310; export TF_CPP_MIN_LOG_LEVEL=3 python dtensor-gpu-gcp/dtensor-app-naive.py"
$ bash cluster-delete.sh
```

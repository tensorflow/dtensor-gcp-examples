# dtensor-gpu-gcp

This project contains an example of using multi-client DTensor on GCP with a
cluster of GPUs.

- dtensor-client.py: the dtensor application. This program will run on each VM
  instance.

- run-multi-nodes.sh: DTensor application launcher. This launcher will launch
  a single instance of the dtensor application on the current VM

- bootstrap-cluster.sh: commands to start the cluster and deploy the application.
  The bootstrap produces a cluster-run.sh. 

- cluster-run.sh: (produced by bootstrap-cluster.sh) runs the proviced command
  on all VMs in the cluster.


Steps to run this application:

```
$ gcloud auth login ...

$ git clone ...
$ cd dtensor-gpu-gcp
$ bash bootstrap-cluster.sh
$ bash cluster-run.sh "cd dtensor-gpu-gcp;bash run-multi-nodes.sh"
```

### multi-client single Node multi-GPU

The project also contains an example launcher that split a VM into
multiple DTensor clients, hiding unwanted GPUs via `CUDA_VISIBILE_GPU` for
each client.

- run-single-node.sh: DTensor application launcher that splits a single node
  into multiple clients. Expects a 4 (physical) GPU node.

# DTensor GCP Examples

This project contains examples of using multi-client DTensor on GCP with a
cluster of GPUs or TPUs.


## Prerequisites

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

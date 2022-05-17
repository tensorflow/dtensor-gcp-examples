import os
import sys


dtensor_jobs = os.environ['DTENSOR_JOBS']

dtensor_jobs = dtensor_jobs.split(',')

if len(sys.argv) == 1:
  hostname = sys.argv[1]
  for i, job in enumerate(dtensor_jobs):
    host, port = job.split(":")
    if host == hostname:
      print(i)
else:
  print(len(dtensor_jobs))

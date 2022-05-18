"""Get client id / client num from DTENSOR_JOBS env.

Syntax:

  get-client.py:
    returns total number of clients (e.g. 4), suitable for DTENSOR_NUM_CLIENTS

  get-client.py hostname: returns client id for clients on host `hostname`, 
    e.g. 0 1 2 3. Suitable for DTENSOR_NUM_CLIENTS. usually use with a for
    loop in bash.
"""
import os
import sys


dtensor_jobs = os.environ['DTENSOR_JOBS']
dtensor_jobs = dtensor_jobs.split(',')

if len(sys.argv) == 2:
  hostname = sys.argv[1]
  for i, job in enumerate(dtensor_jobs):
    host, port = job.split(":")
    if host == hostname:
      print(i)
else:
  print(len(dtensor_jobs))

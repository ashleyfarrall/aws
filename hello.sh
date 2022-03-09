# hello.py:
# usage: python hello.py

from mpi4py import MPI
import sys

def print_hello(rank, size, name):
  msg = "Hello World! I am process {0} of {1} on {2}.\n"
  sys.stdout.write(msg.format(rank, size, name))

if __name__ == "__main__":
  size = MPI.COMM_WORLD.Get_size()
  rank = MPI.COMM_WORLD.Get_rank()
  name = MPI.Get_processor_name()

  print_hello(rank, size, name)
(base) [centos@ip-10-0-0-179 mpitest]$ ^C
(base) [centos@ip-10-0-0-179 mpitest]$ cat helloworld.sh
#!/bin/bash
#SBATCH --job-name=mpi4py-test   # create a name for your job
##SBATCH --nodes=10
#SBATCH --ntasks-per-node=36
##SBATCH --mem=70000         # total memory
##SBATCH --mem-per-cpu=2000
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)

module purge
## module load anaconda3/2021.11
module load openmpi/4.1.1  # REPLACE <x.y.z>
## conda activate fast-mpi4py

mpirun python helloworld.py

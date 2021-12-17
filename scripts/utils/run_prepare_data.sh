#!/bin/bash -l
#SBATCH -D ./

#SBATCH -o I128_prepare.%j
#SBATCH -e I128_prepare.%j
#SBATCH -J I128

### TIME LIMIT: e.g.
#SBATCH --time=0-12:00:00
#SBATCH --signal=USR1@300

### NODE features:
### Num nodes, num tasks per node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

### No need to modify below on raven!
#SBATCH --mem=0
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=72
#SBATCH --threads-per-core=1

### Modules and env variables
source raven.env

### print loaded modules and basic SLURM info
module list

echo -e "Nodes: ${SLURM_JOB_NUM_NODES} \t NTASK: ${SLURM_NTASKS}"
echo "${SLURM_NODELIST}"

python make_hdf5.py --dataset I128 --batch_size 256 --data_root data
python calculate_inception_moments.py --dataset I128_hdf5 --data_root data
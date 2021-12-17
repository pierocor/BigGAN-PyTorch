#!/bin/bash -l
#SBATCH -D ./

#SBATCH -o /ptmp/pierocor/BigGan_out/E128/E128_hvd.%j
#SBATCH -e /ptmp/pierocor/BigGan_out/E128/E128_hvd.%j
#SBATCH -J E128_hvd

### TIME LIMIT: e.g.
### 1-00:00:00 -> 1 day (Maximum)
### 0-00:20:00 -> 20 minutes
#SBATCH --time=1-00:00:00

### NODE features:
### No need to modify them on raven!
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=0
#SBATCH --nodes=8
#SBATCH --ntasks-per-socket=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --threads-per-core=1

### Modules and env variables
source raven.env

### print loaded modules and basic SLURM info
module list

echo -e "Nodes: ${SLURM_JOB_NUM_NODES} \t NTASK: ${SLURM_NTASKS}"
echo "${SLURM_NODELIST}"


DATA_ROOT="/ptmp/pierocor/datasets"  # This should work but you can use a different one
WEIGHTS_ROOT="/ptmp/pierocor/BigGan_out/E128/weights"  # Replace by a folder where you have writing aaccess
LOGS_ROOT="/ptmp/pierocor/BigGan_out/E128/logs"  # Replace by a folder where you have writing aaccess
SAMPLE_ROOT="/ptmp/pierocor/BigGan_out/E128/samples"  # Replace by a folder where you have writing aaccess

### Run the program:
### Change wathever you want and have fun!
srun python train.py \
  --data_root $DATA_ROOT \
  --weights_root $WEIGHTS_ROOT \
  --logs_root $LOGS_ROOT \
  --samples_root $SAMPLE_ROOT \
  --dataset E128_hdf5 --shuffle  --num_workers 0 --batch_size 2048 --copy_in_mem  \
  --num_G_accumulations 1 --num_D_accumulations 1 \
  --num_D_steps 2 --G_lr 5.0e-05 --D_lr 3.0e-04 --D_B2 0.999 --G_B2 0.999 \
  --G_attn 64 --D_attn 64 \
  --G_nl inplace_relu --D_nl inplace_relu \
  --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
  --G_ortho 0.0 \
  --G_shared \
  --G_init ortho --D_init ortho \
  --hier --dim_z 120 --shared_dim 128 \
  --G_eval_mode \
  --G_ch 96 --D_ch 96 \
  --ema --use_ema --ema_start 20000 \
  --test_every 2000 --save_every 1000 --num_best_copies 5 --num_save_copies 2 --seed 0 \
  --num_epochs 350 --resume

  # --num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --D_B2 0.999 --G_B2 0.999 \

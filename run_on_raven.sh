#!/bin/bash -l
#SBATCH -D ./

#SBATCH -o weighting.%j
#SBATCH -e weighting.%j
#SBATCH -J weighting

### TIME LIMIT: e.g.
#SBATCH --time=0-24:00:00
#SBATCH --signal=USR1@300

### NODE features:
### Num nodes, num tasks per node
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=4

### No need to modify below on raven!
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=0
#SBATCH --ntasks-per-socket=2
#SBATCH --cpus-per-task=18
#SBATCH --threads-per-core=1

#SBATCH --mail-type=all
#SBATCH --mail-user=wklos@uos.de

### Modules and env variables
source raven.env

### print loaded modules and basic SLURM info
module list

echo -e "Nodes: ${SLURM_JOB_NUM_NODES} \t NTASK: ${SLURM_NTASKS}"
echo "${SLURM_NODELIST}"


DATA_ROOT="/ptmp/pierocor/datasets"  # This should work but you can use a different directory where the ecoset_cs500.hdf5 is located
### Replace by a folders where you have writing access
WEIGHTS_ROOT="/ptmp/wero/hvd_out/weights"
LOGS_ROOT="/ptmp/wero/hvd_out/logs"
SAMPLE_ROOT="/ptmp/wero/hvd_out/samples"

### Location of the weights of the promising training:
LOAD_FROM="/u/kamue/ecosetbiggan/hvd_out/resume/weights/BigGAN_ecoset_cs500__mem_w0_seed42_Gch96_Dch96_bs1760_hvd40_4_nDs2_Glr5.0e-05_Dlr3.0e-04_Gnlinplace_relu_Dnlinplace_relu_Ginitortho_Dinitortho_Gattn64_Dattn64_Gshared_hier_ema"

### Run the program:
srun python train.py \
  --data_root $DATA_ROOT \
  --weights_root $WEIGHTS_ROOT \
  --logs_root $LOGS_ROOT \
  --samples_root $SAMPLE_ROOT \
  --model BigGAN \
  --class_weights_file ecoset_class_weights.pt \
  --num_epochs 900 \
  --dataset ecoset_cs500 \
  --shuffle  --num_workers 0 --batch_size 1760 \
  --num_G_accumulations 1 --num_D_accumulations 1 \
  --num_D_steps 2 --G_lr 5.0e-05 --D_lr 3.0e-04 --D_B2 0.999 --G_B2 0.999 \
  --G_attn 64 --D_attn 64 \
  --G_nl inplace_relu --D_nl inplace_relu \
  --SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
  --G_ortho 0.0 \
  --G_shared \
  --G_init ortho --D_init ortho \
  --dim_z 140 --shared_dim 128 --hier \
  --G_eval_mode \
  --G_ch 96 --D_ch 96 \
  --ema --use_ema --ema_start 20000 \
  --test_every 1000 --save_every 1000 \
  --num_best_copies 5 --num_save_copies 2 \
  --copy_in_mem \
  # --hier --seed 42 --resume \
  # --load_from ${LOAD_FROM} --load_weights best4

### WARNING! Run the script as it is just once. After that, to resume training remove the --load_from flag

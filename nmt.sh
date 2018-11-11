#!/bin/bash
#BATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:k80:1
#SBATCH --time=15:00:00
#SBATCH --mem=6GB
#SBATCH --job-name=migatte_no_gokui
#SBATCH --mail-type=END
#SBATCH --mail-user=jm7432@nyu.edu
#SBATCH --output=slurm_%j.out

module purge

SRCDIR=$HOME
RUNDIR=$SCRATCH/nmt/rnn/
mkdir -p $RUNDIR

cd $SLURM_SUBMIT_DIR
cp -r $SRCDIR/Neural-Machine-Translation $RUNDIR

cd $RUNDIR
module load python3/intel/3.6.3
module load cuda/9.0.176

python3 ./Neural-Machine-Translation/main.py --output exp1_
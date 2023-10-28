#!/bin/bash
#SBATCH --account=def-sponsor00
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1

module load nixpkgs/16.09
module load spark/2.3.0

# Recommended settings for calling Intel MKL routines from multi-threaded applications
# https://software.intel.com/en-us/articles/recommended-settings-for-calling-intel-mkl-routines-from-multi-threaded-applications 
export MKL_NUM_THREADS=1
export SPARK_IDENT_STRING=$SLURM_JOBID
export SPARK_WORKER_DIR=$SLURM_TMPDIR
export SLURM_SPARK_MEM=$(printf "%.0f" $((${SLURM_MEM_PER_NODE} *95/100)))

start-master.sh
sleep 5
MASTER_URL=$(grep -Po '(?=spark://).*' $SPARK_LOG_DIR/spark-${SPARK_IDENT_STRING}-org.apache.spark.deploy.master*.out)

NWORKERS=$((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES - 1))
SPARK_NO_DAEMONIZE=1 srun -n ${NWORKERS} -N ${NWORKERS} --label --output=$SPARK_LOG_DIR/spark-%j-workers.out start-slave.sh -m ${SLURM_SPARK_MEM}M -c ${SLURM_CPUS_PER_TASK} ${MASTER_URL} &
slaves_pid=$!

srun -n 1 -N 1 spark-submit mltree_cluster.py ${SLURM_JOB_NUM_NODES} ${SLURM_CPUS_PER_TASK} --master ${MASTER_URL} --executor-memory ${SLURM_SPARK_MEM}M

kill $slaves_pid
stop-master.sh

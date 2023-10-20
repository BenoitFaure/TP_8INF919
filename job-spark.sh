#!/bin/bash
#SBATCH --account=def-sponsor00
#SBATCH --time=00:30:00
#SBATCH --nodes=10
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH –mail-user=benoit.faure.cesar@gmail.com

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

SLURM_SPARK_SUBMIT="srun -n 1 -N 1 spark-submit --master ${MASTER_URL} ml_tree.py  --executor-memory ${SLURM_SPARK_MEM}M"
$SLURM_SPARK_SUBMIT --class org.apache.spark.examples.SparkPi $SPARK_HOME/examples/jars/spark-examples_2.11-2.3.0.jar 1000
$SLURM_SPARK_SUBMIT --class org.apache.spark.examples.SparkLR $SPARK_HOME/examples/jars/spark-examples_2.11-2.3.0.jar 1000

kill $slaves_pid
stop-master.sh

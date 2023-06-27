#!/bin/bash
#SBATCH --partition compute           # Submit to 'compute' Partition or queue
#SBATCH --job-name=multiatlasprop
#SBATCH --output=logs/run_multiatlasprop_array_%A_%a.out
#SBATCH -e logs/run_multiatlasprop_array_%A_%a.err
#SBATCH --time=0-72:00:00        # Run for a maximum time of 0 days, 72 hours, 00 mins, 00 secs
#SBATCH --nodes=1            # Request N nodes
#SBATCH --ntasks-per-node=64  # Request n cores or task per node
#SBATCH --mem-per-cpu=3500M

# the environment variable SLURM_ARRAY_TASK_ID contains
# the index corresponding to the current job step

#echo ------ $1
#./runGIF_array_program.sh "$@"

# string inputs
#imag_paths_str=${1}
#mask_paths_str=${2}
#out_paths_str=${3}
#cpp_paths_str=${4}
#nb_of_submitted_jobs_limit=${5}

# parse string inputs into arrays
#IFS=' ' read -ra imag_paths <<< "$imag_paths_str"
#IFS=' ' read -ra mask_paths <<< "$mask_paths_str"
#IFS=' ' read -ra out_paths <<< "$out_paths_str"
#IFS=' ' read -ra cpp_paths <<< "$cpp_paths_str"

# select the correct paths based on SLURM_ARRAY_TASK_ID
#imag_path=${imag_paths[$SLURM_ARRAY_TASK_ID]}
#mask_path=${mask_paths[$SLURM_ARRAY_TASK_ID]}
#out_path=${out_paths[$SLURM_ARRAY_TASK_ID]}
#cpp_path=${cpp_paths[$SLURM_ARRAY_TASK_ID]}


module load miniconda
conda activate local

echo Submitting run_multi_atalas_segmentation_test.py

timestamp=$(date -d "today" +"%Y-%m-%d-%H_%M_%S")
python -m cProfile -o logs/profile_${timestamp}.dat run_multi_atalas_segmentation_test.py
#python run_multi_atalas_segmentation_test.py

cp logs/profile_${timestamp}.dat logs/profile_latest.dat

#/home/aku20/tools/GIF/install/bin/seg_GIF \
#-in $imag_path \
#-mask $mask_path \
#-db /home/aku20/tools/GIF/db_masked/db.xml \
#-out $out_path \
#-cpp $cpp_path \
#-v 1 \
#-omp 64 \
#-ompj 64 \
#-lncc_ker -3 \
#-temper 0.05 \
#-regNMI \
#-regBE 0.001 \
#-regJL 0.00005 \
#-geo

#if [[ $(((SLURM_ARRAY_TASK_ID+1) % nb_of_submitted_jobs_limit)) == 0 ]] ; then 
#    sbatch --array=$((SLURM_ARRAY_TASK_ID+1))-$((SLURM_ARRAY_TASK_ID+nb_of_submitted_jobs_limit)) "$0" "$@"
#fi

# nb_cases=${#imag_paths[@]}
# next_task_id=$((SLURM_ARRAY_TASK_ID+nb_of_submitted_jobs_limit))
# # submit job from next batch of array jobs
# if (( next_task_id < nb_cases )); then
# 	sbatch --array=$next_task_id "$0" "$@"
# fi

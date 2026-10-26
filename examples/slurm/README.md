# SLURM examples

Scripts used to run the pipeline as a SLURM job array on one specific cluster (partition `compute`,
`module load miniconda`, conda environment `local`). They are kept as examples; adapt the `#SBATCH`
lines and the environment set-up to your site.

* `tom2_submit_parallel_script.py PYTHON_SCRIPT NB_CASES` submits `tom2_sbatch_script_taskid.sh` as a job
  array; each task runs `python PYTHON_SCRIPT --taskid <array index>`. The scripts in
  `scripts/run_multi_atlas_segmentation/` accept `--taskid`.
* `tom2_sbatch_script.sh` runs an arbitrary command under the Python profiler.

Create the `logs/` folder before submitting, because the job output is written to it.

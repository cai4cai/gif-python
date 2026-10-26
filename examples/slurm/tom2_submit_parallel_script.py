"""Submit a SLURM job array that runs a script once per task id.

Usage::

    python examples/slurm/tom2_submit_parallel_script.py PYTHON_SCRIPT NB_CASES

``PYTHON_SCRIPT`` must accept ``--taskid N`` (see ``scripts/run_multi_atlas_segmentation``). The
partition, time limit, modules and conda environment in ``tom2_sbatch_script_taskid.sh`` are specific
to the cluster this was written for; adapt them to yours.
"""

import subprocess
import sys
from pathlib import Path

sbatch_script_path = Path(__file__).resolve().with_name("tom2_sbatch_script_taskid.sh")
python_script_path = sys.argv[1]  # path to the python script to be executed (takes 1 argument: the task id --taskid )
nb_cases = int(sys.argv[2])  # total number of taskid to be submitted

nb_of_active_jobs = 8
nb_of_submitted_jobs_limit = 99999


nb_jobs_to_submit_at_once = min(nb_of_submitted_jobs_limit, nb_cases)
array_indeces_str = f"0-{nb_jobs_to_submit_at_once - 1}%{nb_of_active_jobs}"

print(f"Number of cases is {nb_cases}")
print(f"array_indeces_str is {array_indeces_str}")

sbatch_command = [
    "sbatch",
    "-a", array_indeces_str,
    str(sbatch_script_path),
    python_script_path,
    str(nb_of_submitted_jobs_limit),
]  # fmt: skip
print("\n\nsbatch_command =", " ".join(sbatch_command))
subprocess.run(sbatch_command, check=True)

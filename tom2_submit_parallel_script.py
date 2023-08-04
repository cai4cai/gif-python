import os
from glob import glob
from natsort import natsorted
import sys

sbatch_script_path = sys.argv[1]

nb_of_active_jobs = 8
nb_of_submitted_jobs_limit = 99999

atlas_paths = [d for d in natsorted(glob("../data/atlases/NMM_atlases/*")) if os.path.isdir(d)]
atlas_names = [os.path.basename(atlas_path) for atlas_path in atlas_paths]

nb_cases = len(atlas_paths)
nb_jobs_to_submit_at_once = min(nb_of_submitted_jobs_limit, nb_cases)
array_indeces_str = f"0-{nb_jobs_to_submit_at_once - 1}%{nb_of_active_jobs}"

print(f"Number of cases is {nb_cases}")
print(f"array_indeces_str is {array_indeces_str}")

atlas_names_str = " ".join(atlas_names)

# wrap atlas_names_str in double quotes
atlas_names_str = f'"{atlas_names_str}"'


print("\natlas_names_str =", atlas_names_str)

# Replace the following line with the actual command for submitting the job.
sbatch_command = f"sbatch -a {array_indeces_str} {sbatch_script_path} {atlas_names_str} {str(nb_of_submitted_jobs_limit)}"
print("\n\nsbatch_command =", sbatch_command)
os.system(sbatch_command)


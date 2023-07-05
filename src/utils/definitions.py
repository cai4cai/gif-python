import os

# REGISTRATION HYPER-PARAMETERS
GRID_SPACING = 4  # in mm (default is 4 mm = 5 voxels x 0.8 mm.voxels**(-1))
BE = 0.005
LE = 0.001  # Weight of first order penalty term (symmetric and anti-symmetric part of the Jacobian)
LP = 3  # default 3; we do only the lp first level of the pyramid

# RESAMPLING
RESAMPLE_METHOD = 0  # 0: nearest neighbor interpolation on label map, 1: (smoothing and) linear interpolation on one-hot encoded labels

# MULTIPROCESSING
MULTIPROCESSING = True

# PARENT FOLDERS
HOME_FOLDER = '/home/aaron'
WORKSPACE_FOLDER = os.path.join(HOME_FOLDER, 'Dropbox/KCL/Projects')

# NIFTYREG_PATH = os.path.join(WORKSPACE_FOLDER, 'third-party', 'niftyreg', 'build', 'reg-apps')
NIFTYREG_PATH = os.path.join(
    WORKSPACE_FOLDER, 'trustworthy-ai-fetal-brain-segmentation',
    'docker', 'third-party', 'niftyreg', 'build', 'reg-apps',
)
NIFTYSEG_PATH = "/usr/local/bin"



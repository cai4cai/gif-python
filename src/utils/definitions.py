import os

# LINEAR REGISTRATION HYPER-PARAMETERS
reg_aladin_LP = 2  # Number of levels to use to run the registration once the pyramids have been created.

# NONLINEAR REGISTRATION HYPER-PARAMETERS
reg_f3d_JL = 0.0001  # Weight of log of the Jacobian determinant penalty term
reg_f3d_BE = 0.005  # Weight of the bending energy (second derivative of the transformation) penalty term
reg_f3d_MAXIT = 100  # Maximal number of iteration at the final level
reg_f3d_LN = 4  # Number of level to perform
reg_f3d_LP = 3  # default 3; we do only the lp first level of the pyramid
reg_f3d_GRID_SPACING = -5.0  # Final grid spacing along the x, y, z axes in mm (in voxel if negative value), corresponds to -sx option in reg_f3d
reg_f3d_LNCC = 5.0  # Standard deviation of the Gaussian kernel.
reg_f3d_INTERP = 1  # Interpolation order (0=NN, 1=linear, 3=cubic)

# RESAMPLING
RESAMPLE_METHOD = 0  # 0: nearest neighbor interpolation on label map, 1: (smoothing and) linear interpolation on one-hot encoded labels

# RESAMPLING HYPER-PARAMETERS
SIGMA = 0  # Standard deviation of the Gaussian kernel used to smooth the label maps before resampling, only applies to RESAMPLE_METHOD = 1

# MULTIPROCESSING FOR LOOP OVER ATLASES
MULTIPROCESSING = True
NUM_POOLS = 14

# IN EACH OF THE ABOVE PROCESSES, HOW MANY SUBPROCESSES should reg_aladin and reg_f3d run in parallel?
OMP = 8

# PARENT FOLDERS
HOME_FOLDER = '/home/aaron'
WORKSPACE_FOLDER = os.path.join(HOME_FOLDER, 'Dropbox/KCL/Projects')

# NIFTYREG_PATH = os.path.join(WORKSPACE_FOLDER, 'third-party', 'niftyreg', 'build', 'reg-apps')
# NIFTYREG_PATH = os.path.join(
#     WORKSPACE_FOLDER, 'trustworthy-ai-fetal-brain-segmentation',
#     'docker', 'third-party', 'niftyreg', 'build', 'reg-apps',
# )
NIFTYREG_PATH = "/usr/local/bin"
NIFTYSEG_PATH = "/usr/local/bin"



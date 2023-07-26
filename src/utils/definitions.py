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

# WEIGHTS CALCULATION
WEIGHTS_TEMPERATURE = 0.15  # Temperature parameter for the conversion from LNCC distance to weights

# WHETHER To USE PREVIOUS RESULTS OF REGISTRATION, RESAMPLING, AND WEIGHTS CALCULATION
USE_OLD_RESULTS = False

# MULTIPROCESSING FOR LOOP OVER ATLASES
MULTIPROCESSING = True
NUM_POOLS = 14  # More pools will require more RAM, be careful not to exceed the available RAM

# IN EACH OF THE ABOVE PROCESSES, HOW MANY SUBPROCESSES should reg_aladin and reg_f3d run in parallel?
OMP = 8  # OMP*NUM_POOLS should not exceed the number of cores available

# NIFTYREG AND NIFTYSEG PATHS
NIFTYREG_PATH = "/usr/local/bin"  # should contain reg_aladin, reg_f3d, reg_resample binaries
NIFTYSEG_PATH = "/usr/local/bin"  # should contain seg_EM binary



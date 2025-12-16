# Installation Guide

This guide provides detailed instructions for installing all the required software and dependencies for this project.

## Table of Contents
- [Python Installation](#python-installation)
- [NiftyReg Installation](#niftyreg-installation)
- [NiftySeg Installation](#niftyseg-installation)
- [Python Libraries](#python-libraries)
- [Data Folder Setup](#data-folder-setup)
- [Verification](#verification)

## Python Installation

### Requirement
Python 3.8 or higher is required.

### Check Current Python Version
```bash
python --version
# or
python3 --version
```

### Installation Options

#### macOS
Using Homebrew:
```bash
brew install python@3.8
```

Or download from [python.org](https://www.python.org/downloads/)

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.8 python3-pip
```

#### Linux (Fedora/CentOS)
```bash
sudo dnf install python38
```

#### Windows
Download and install from [python.org](https://www.python.org/downloads/)

Make sure to check "Add Python to PATH" during installation.

## NiftyReg Installation

NiftyReg is a medical image registration library.

### From Source (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/KCL-BMEIS/niftyreg.git
cd niftyreg
```

2. Build and install:
```bash
mkdir build
cd build
cmake ..
make
sudo make install
```

### Prerequisites
NiftyReg requires CMake and a C++ compiler:

**macOS:**
```bash
brew install cmake
```

**Linux:**
```bash
sudo apt install cmake build-essential  # Ubuntu/Debian
sudo dnf install cmake gcc-c++          # Fedora/CentOS
```

**Windows:**
Install CMake from [cmake.org](https://cmake.org/download/) and Visual Studio Build Tools.

### Verify Installation
```bash
reg_aladin --version
```

## NiftySeg Installation

NiftySeg is a medical image segmentation library.

### From Source (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/KCL-BMEIS/niftyseg.git
cd niftyseg
```

2. Build and install:
```bash
mkdir build
cd build
cmake ..
make
sudo make install
```

### Verify Installation
```bash
seg_EM --version
```

## Python Libraries

The following Python libraries are required:
- numpy
- nibabel
- scipy
- pandas
- numba

### Installation

Install all required libraries at once:
```bash
pip install numpy nibabel scipy pandas numba
```

Or install individually:
```bash
pip install numpy
pip install nibabel
pip install scipy
pip install pandas
pip install numba
```

### Using a Virtual Environment (Recommended)

It's recommended to use a virtual environment to avoid conflicts with other Python projects:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install the required libraries
pip install numpy nibabel scipy pandas numba
```

## Data Folder Setup

The `data/` folder is required for running multi-atlas segmentation and is **not included** in this repository. You need to set it up with the appropriate atlas data.

### Expected Data Folder Structure

```
data/
├── atlases/
│   ├── Mindboggle101/                  # Mindboggle atlases
│   │   ├── structures_info.csv         # Label definitions
│   │   ├── tissues_info.csv            # Tissue type definitions
│   │   ├── OASIS-TRT-20-1/
│   │   │   ├── t1weighted.nii.gz
│   │   │   ├── labels.DKT31.manual+aseg.nii.gz
│   │   │   └── labels.DKT31.manual+aseg_cleaned.nii.gz
│   │   ├── OASIS-TRT-20-2/
│   │   │   ├── t1weighted.nii.gz
│   │   │   ├── labels.DKT31.manual+aseg.nii.gz
│   │   │   └── labels.DKT31.manual+aseg_cleaned.nii.gz
│   │   └── ...                         # (101 subjects total)
│
└── input/                              # Your subjects to segment (optional)
    └── subject-01/
        └── t1weighted_brain.nii.gz
```

### Required Files per Atlas Subject

Each atlas directory must contain:
- **`t1weighted.nii.gz`**: T1-weighted MRI image
- **`labels.DKT31.manual+aseg.nii.gz`**: Manual segmentation labels (DKT31 parcellation)
- **`labels.DKT31.manual+aseg_cleaned.nii.gz`**: Manual segmentation labels (DKT31 parcellation) without the labels [1001, 1032, 1033, 2001, 2032, 2033] after having used clean_small_labels.py

### Required CSV Files

#### `structures_info.csv`
Defines label IDs, names, and tissue classifications:
```csv
label,name,tissues
0,Unknown,[0]
2,Left-Cerebral-White-Matter,[3]
1002,ctx-lh-caudalanteriorcingulate,[2]
...
```

#### `tissues_info.csv`
Defines tissue type IDs and names:
```csv
label,name
0,Non-Brain Outer Tissue
1,Cerebral Spinal Fluid
2,Grey Matter
3,White Matter
4,Deep Grey Matter
5,Brain Stem and Pons
```

### Obtaining Atlas Data

The Mindboggle101 atlas data can be obtained from:

1. **Synapse** (Recommended): [https://www.synapse.org/Synapse:syn3218329/files/](https://www.synapse.org/Synapse:syn3218329/files/)
2. **Mindboggle Website**: [https://mindboggle.info/data.html](https://mindboggle.info/data.html)

After downloading:
1. Extract the atlas data
2. Organize it according to the structure above
3. Ensure each subject has both the T1-weighted image and labels
4. Place the `structures_info.csv` and `tissues_info.csv` files in the atlas root directory


## Verification

After installation, verify that all components are properly installed:

### Check Python Libraries
```python
python -c "import numpy, nibabel, scipy, pandas, numba; print('All libraries imported successfully')"
```

### Check NiftyReg
```bash
which reg_aladin
reg_aladin --version
```

### Check NiftySeg
```bash
which seg_EM
seg_EM --version
```

## Troubleshooting

### Library Import Errors
If you encounter import errors, ensure pip is up to date:
```bash
pip install --upgrade pip
```

### Permission Issues
If you encounter permission errors during installation, you may need to use `sudo` (Linux/macOS) or run as administrator (Windows).

For pip installations without sudo, use the `--user` flag:
```bash
pip install --user numpy nibabel scipy pandas numba
```

### NiftyReg/NiftySeg Build Errors
Ensure all build dependencies (CMake, compiler) are properly installed. Check the respective GitHub repositories for platform-specific build instructions and known issues.

## Additional Resources

- [NiftyReg Documentation](https://github.com/KCL-BMEIS/niftyreg)
- [NiftySeg Documentation](https://github.com/KCL-BMEIS/niftyseg)
- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)

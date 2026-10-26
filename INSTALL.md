# Installation Guide

This guide provides detailed instructions for installing all the required software and dependencies for this project.

## Table of Contents
- [Python Installation](#python-installation)
- [NiftyReg Installation](#niftyreg-installation)
- [NiftySeg Installation](#niftyseg-installation)
- [Python Libraries](#python-libraries)
- [Locating NiftyReg and NiftySeg](#locating-niftyreg-and-niftyseg)
- [Data Folder Setup](#data-folder-setup)
- [Verification](#verification)

## Python Installation

### Requirement
Python 3.9 or higher is required.

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
brew install python@3.12  # any version >= 3.9
```

Or download from [python.org](https://www.python.org/downloads/)

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv   # needs to be 3.9 or newer
```

#### Linux (Fedora/CentOS)
```bash
sudo dnf install python3 python3-pip
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

The Python dependencies (numpy, scipy, nibabel, pandas, numba, natsort, tqdm, xmltodict) are declared in
`pyproject.toml` and installed with the package:

```bash
python -m venv venv && source venv/bin/activate     # recommended; on Windows: venv\Scripts\activate
pip install .                                       # or: pip install -e ".[dev]" for development
```

Check:

```bash
gif-python --help
python -c "import src; print(src.__version__)"
```

## Locating NiftyReg and NiftySeg

`reg_aladin`, `reg_f3d`, `reg_resample` (NiftyReg) and `seg_EM` (NiftySeg) are searched in this order:
the directory given in the configuration (`--niftyreg-path` / `--niftyseg-path`, `GifConfig(niftyreg_path=...)`),
the directories in the environment variables `NIFTYREG_PATH` and `NIFTYSEG_PATH`, `PATH`, and `/usr/local/bin`.
A missing executable is reported with the name and the places that were searched.

```bash
export NIFTYREG_PATH=$HOME/niftyreg/install/bin
export NIFTYSEG_PATH=$HOME/niftyseg/install/bin
gif-python segment --print-config | tail -2     # shows the directories that will be used
```

The refactoring of this package was checked with NiftyReg 1.5.77.

## Data Folder Setup

The `data/` folder is **not included** in this repository. The folder layout used in the documentation is:

```
data/
├── atlases/
│   └── Mindboggle101/
│       ├── structures_info.csv                  # label definitions
│       ├── tissues_info.csv                     # tissue definitions
│       ├── itksnap_label_descriptions_mindboggle101.txt
│       ├── OASIS-TRT-20-1/
│       │   ├── t1weighted.nii.gz
│       │   ├── t1weighted_brain.nii.gz
│       │   ├── labels.DKT31.manual+aseg.nii.gz
│       │   └── labels.DKT31.manual+aseg_cleaned.nii.gz
│       └── ...                                  # 101 subjects
└── downloads/mindboggle101/                     # the downloaded archives
```

For Mindboggle-101 this is created by one command (about 3.3 GB, no login):

```bash
gif-python fetch-mindboggle --clean
```

It downloads the five volume archives from Harvard Dataverse, verifies their SHA-256, extracts the files above,
copies the csv files, and (with `--clean`) creates the `_cleaned` labels, i.e. the labels without
[1001, 1032, 1033, 2001, 2032, 2033]. See [docs/mindboggle101.md](docs/mindboggle101.md).

### Required files per atlas

Each atlas folder contains an image and a segmentation (names are options of `gif-python segment`:
`--atlas-img`, `--atlas-seg`, optionally `--atlas-mask`).

### `structures_info.csv`

Defines label ids, names, and the tissue classes each structure can belong to:

```csv
label,name,tissues
0,Unknown,[0]
2,Left-Cerebral-White-Matter,[3]
1002,ctx-lh-caudalanteriorcingulate,[2]
...
```

### `tissues_info.csv`

Defines tissue type ids and names:

```csv
label,name
0,Non-Brain Outer Tissue
1,Cerebral Spinal Fluid
2,Grey Matter
3,White Matter
4,Deep Grey Matter
5,Brain Stem and Pons
```

For other atlases, `gif-python make-structure-info` generates both files from the atlas segmentations and a
FreeSurfer colour table.

## Verification

```bash
which reg_aladin reg_f3d reg_resample seg_EM      # or check $NIFTYREG_PATH / $NIFTYSEG_PATH
reg_aladin --version
seg_EM --version
pytest                                            # tests marked `integration` run when the tools are found
```

## Troubleshooting

### Library Import Errors
If you encounter import errors, ensure pip is up to date:
```bash
pip install --upgrade pip
```

### Permission Issues
Install into a virtual environment (see above) instead of the system Python. To install NiftyReg/NiftySeg
without `sudo`, use `cmake -DCMAKE_INSTALL_PREFIX=$HOME/niftyreg/install ..` and set `NIFTYREG_PATH`
(and likewise for NiftySeg).

### NiftyReg/NiftySeg Build Errors
Ensure all build dependencies (CMake, compiler) are properly installed. Check the respective GitHub repositories for platform-specific build instructions and known issues.

## Additional Resources

- [NiftyReg Documentation](https://github.com/KCL-BMEIS/niftyreg)
- [NiftySeg Documentation](https://github.com/KCL-BMEIS/niftyseg)
- [Python Virtual Environments Guide](https://docs.python.org/3/tutorial/venv.html)

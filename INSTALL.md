# Installation Guide

This guide provides detailed instructions for installing all the required software and dependencies for this project.

## Table of Contents
- [Python Installation](#python-installation)
- [NiftyReg Installation](#niftyreg-installation)
- [NiftySeg Installation](#niftyseg-installation)
- [Python Libraries](#python-libraries)
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

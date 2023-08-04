from setuptools import setup, find_packages

setup(
    name='pygif',
    version='1.0.0',
    url='https://github.com/aaronkujawa/gif-python.git',
    author='Aaron Kujawa',
    author_email='askujawa@gmail.com',
    description='Python implementation of GIF',
    packages=find_packages(),    
    install_requires=['numpy >= 1.11.1'],
)

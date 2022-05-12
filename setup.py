# setup.py
# setup script for mrae

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

install_requires = [
    'torch',
    'numpy',
    'h5py',
    'tqdm'
]

setuptools.setup(
    name="mrae",
    version="0.1.0",
    author="Michael Nolan",
    author_email="mnolan@uw.edu", # I am not eternal, please replace ~ MN
    description="pytorch implementation of the MRAE time series reconstruction model",
    long_description=long_description,
    url='',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"
    ],
    install_requires=install_requires,
)

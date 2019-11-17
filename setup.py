import setuptools
from shutil import copyfile

with open("README.md", "r") as fh:
    long_description = fh.read()

#make copy of pysnn
copyfile("bin/pysnn", "bin/PYSNN")

setuptools.setup(
    name="PYSNN",
    version="0.1.2",
    author="Lukáš Plevač",
    author_email="lukasplevac@gmail.com",
    description="PYthon Simple Neural Network - PYSNN is python3 lib for machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Lukas0025/PYSNN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    scripts=[
        'bin/pysnn',
        'bin/PYSNN'
    ],
    python_requires='>=3.0',
    install_requires=[
        'numpy',
	    'requests'
    ],
)

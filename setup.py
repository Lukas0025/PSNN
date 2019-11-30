import setuptools
from shutil import copyfile

with open("README.md", "r") as fh:
    long_description = fh.read()

#make copy of PSNN
copyfile("bin/psnn", "bin/PSNN")

setuptools.setup(
    name="PSNN",
    version="0.1.5",
    author="Lukáš Plevač",
    author_email="lukasplevac@gmail.com",
    description="Python Simple Neural Network - PSNN is python3 lib for machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Lukas0025/PSNN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    scripts=[
        'bin/psnn',
        'bin/PSNN'
    ],
    python_requires='>=3.0',
    install_requires=[
        'numpy',
	    'requests'
    ],
)

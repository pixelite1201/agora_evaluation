#------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2021 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
#------------------------------------------------------------------------------

"""Script for packaging the project."""

from pathlib import Path
from setuptools import setup, find_packages

PROJECT_NAME = "agora_evaluation"


def _get_version():
    """"Utility function to get the version of this package."""

    ns = {}
    version_path = Path(PROJECT_NAME) / "version.py"
    if not version_path.is_file():
        return
    with open(version_path) as version_file:
        exec(version_file.read(), ns)
    return ns["__version__"]


dependencies = (
    "chumpy",
    "matplotlib",
    "opencv-python==4.2.0.34",
    "pandas==1.0.3",
    "seaborn",
    "sklearn",
    # "smplx", # installed as submodule
    "torch==1.4.0",
    "trimesh",
    "tqdm",
)


setup(
    name=PROJECT_NAME,
    version=_get_version(),
    packages=find_packages(),
    install_requires=dependencies,
    description="Agora evaluation algorithm",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    platforms=["Linux"],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': ['evaluate_agora=agora_evaluation.cli:evaluate_agora', 'project_joints=agora_evaluation.cli:project_joints']
    }
)

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
"""Command-line interface."""
import sys

from agora_evaluation.evaluate_agora import run_evaluation
from agora_evaluation.project_joints import run_projection

def project_joints():
    """Executable for running the evaluation algorithm."""
    run_projection(sys.argv[1:])

def evaluate_agora():
    """Executable for running the evaluation algorithm."""
    run_evaluation(sys.argv[1:])

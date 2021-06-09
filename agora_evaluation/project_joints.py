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
import argparse
import logging
import os
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas
import pickle

from .get_joints_verts_from_dataframe import add_joints_verts_in_dataframe, get_pred_joints

def run_projection(*args):
    """Function to run the evaluation."""

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelFolder', type=str,
                        default='demo/model/smplx')
    parser.add_argument('--debug_path', type=str,
                        default='')
    parser.add_argument('--numBetas', type=int, default=10)
    parser.add_argument('--imgHeight', type=int, default=2160,
                        help='Height of the image')
    parser.add_argument('--imgWidth', type=int, default=3840,
                        help='Width of the image')
    parser.add_argument(
        '--imgFolder',
        type=str,
        default='',
        help='Path to the folder containing test/validation images')
    parser.add_argument('--loadPrecomputed', type=str, default='',
                        help='Path to the ground truth SMPL/SMPLX dataframe')
    parser.add_argument(
        '--modeltype',
        type=str,
        default='SMPLX',
        help='SMPL or SMPLX')
    parser.add_argument('--kid_template_path', type=str, default='template')
    parser.add_argument('--gt_model_path', type=str, default='')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args(*args)
    #Because AGORA pose params are in vector format
    args.pose2rot = True 
    imgHeight = args.imgHeight
    imgWidth = args.imgWidth

    all_df = glob(os.path.join(args.loadPrecomputed, '*.pkl'))
    for df_iter, df_path in tqdm(enumerate(all_df)):
        logging.info(
            'Processing {}th dataframe'.format(
                str(df_iter)))
        df = pandas.read_pickle(df_path)
        # Check if gt joints and verts are stored in dataframe. If not
        # generate them ####
        if 'gt_joints_2d' not in df or 'gt_joints_3d' not in df:
            logging.info('Generating Ground truth joints')
            store_joints = True
            df = add_joints_verts_in_dataframe(args, df, store_joints)
            # write new dataframe with joints and verts stored
            df.to_pickle(df_path.replace('.pkl', '_withjv.pkl'))
        else:
            raise KeyError('Joints already stored in the dataframe. Please remove it before processing')

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

from . import calculate_v2v_error as v2v
from . import load_predictions
from .compute_average_error import compute_avg_error
from .get_joints_verts_from_dataframe import add_joints_verts_in_dataframe, get_pred_joints
from .matching import get_matching


logging.basicConfig(level=logging.DEBUG)
matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(level=logging.WARNING)


def run_evaluation(*args):
    """Function to run the evaluation."""

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', type=str, default=None,
                        help='Path containing the predicitons')
    parser.add_argument('--debug_path', type=str, default=None,
                        help='Path where the debug files will be stored')
    parser.add_argument('--modelFolder', type=str,
                        default='demo/model/smplx')
    parser.add_argument('--numBetas', type=int, default=10)

    parser.add_argument('--result_savePath', type=str, default=None,
                        help='Path where all the results will be saved')
    parser.add_argument(
        '--indices_path',
        type=str,
        default='',
        help='Path to hand,face and body vertex indices for SMPL-X')
    parser.add_argument('--imgHeight', type=int, default=2160,
                        help='Height of the image')
    parser.add_argument('--imgWidth', type=int, default=3840,
                        help='Width of the image')
    parser.add_argument('--numBodyJoints', type=int, default=24,
                        help='Num of body joints used for evaluation')
    parser.add_argument(
        '--imgFolder',
        type=str,
        default='',
        help='Path to the folder containing test/validation images')
    parser.add_argument('--loadPrecomputed', type=str, default='',
                        help='Path to the ground truth SMPL/SMPLX dataframe')
    parser.add_argument(
        '--loadMatched',
        type=str,
        default='',
        help='Path to the dataframe after the matching is done')
    parser.add_argument(
        '--meanPoseBaseline',
        default=False,
        action='store_true')
    parser.add_argument(
        '--onlyComputeErrorLoadPath',
        type=str,
        default='',
        help='Path to the dataframe with all the errors already calculated and stored')
    parser.add_argument(
        '--baseline',
        type=str,
        default='SPIN',
        help='Name of the baseline or the model being evaluated')
    parser.add_argument(
        '--modeltype',
        type=str,
        default='SMPLX',
        help='SMPL or SMPLX')
    parser.add_argument('--kid_template_path', type=str, default='template')
    parser.add_argument('--gt_model_path', type=str, default='')
    parser.add_argument('--onlybfh', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args(*args)
    imgHeight = args.imgHeight
    imgWidth = args.imgWidth

    error_df_list = []
    # If only average error needs to be computed then one can provide the path to the dataframe in
    # onlyComputeErrorLoadPath flag
    if args.onlyComputeErrorLoadPath == '':
        all_df = glob(os.path.join(args.loadPrecomputed, '*.pkl'))
        for df_iter, df_path in tqdm(enumerate(all_df)):
            logging.info(
                'Processing {}th dataframe'.format(
                    str(df_iter)))
            df = pandas.read_pickle(df_path)

            ##################### get prediction joints###############
            logging.info('Loading predictions')
            
            df.insert(column='matching', loc=len(df.columns), value=len(df) * [''])
            df = load_predictions.load(args, df)
            df = get_matching(args, df, imgWidth, imgHeight)
            if not os.path.exists(args.result_savePath):
                os.makedirs(args.result_savePath)

            ############################## compute error for matched and add to the dataframe ##########
            df.insert(
                column='body-MPJPE',
                loc=len(
                    df.columns),
                value=len(df) *
                [''])
            df.insert(
                column='falsePositives',
                loc=len(
                    df.columns),
                value=len(df) *
                [''])
            df.insert(
                column='misses',
                loc=len(
                    df.columns),
                value=len(df) *
                [''])
            df.insert(
                column='miss_flag',
                loc=len(
                    df.columns),
                value=len(df) *
                [''])
            df.insert(
                column='body-MVE',
                loc=len(
                    df.columns),
                value=len(df) *
                [''])

            if args.modeltype == 'SMPLX':
                df.insert(
                    column='lhand-MPJPE',
                    loc=len(
                        df.columns),
                    value=len(df) *
                    [''])
                df.insert(
                    column='rhand-MPJPE',
                    loc=len(
                        df.columns),
                    value=len(df) *
                    [''])
                df.insert(
                    column='face-MPJPE',
                    loc=len(
                        df.columns),
                    value=len(df) *
                    [''])
                df.insert(
                    column='lhand-MVE',
                    loc=len(
                        df.columns),
                    value=len(df) *
                    [''])
                df.insert(
                    column='rhand-MVE',
                    loc=len(
                        df.columns),
                    value=len(df) *
                    [''])
                df.insert(
                    column='face-MVE',
                    loc=len(
                        df.columns),
                    value=len(df) *
                    [''])


            #Load vertex indices to calculate verts error
            verts_ind_dict = {}
            verts_ind_dict['body-smpl'] = np.load(
                os.path.join(
                    args.indices_path,
                    'body_verts_smpl.npy'))
            verts_ind_dict['body-smplx'] = np.load(
                os.path.join(args.indices_path, 'body_verts_smplx.npy'))
            verts_ind_dict['face'] = np.load(
                os.path.join(
                    args.indices_path,
                    'SMPL-X__FLAME_vertex_ids.npy'))
            hands = pickle.load(
                open(
                    os.path.join(
                        args.indices_path,
                        'MANO_SMPLX_vertex_ids.pkl'),
                    'rb'))
            verts_ind_dict['left_hand'] = hands['left_hand']
            verts_ind_dict['right_hand'] = hands['right_hand']


            for idx in tqdm(range(len(df))):
                predJoints = []
                predVerts = []
                allJointsPred = []
                validGts = []
                df.iloc[idx, df.columns.get_loc('body-MPJPE')] = []
                df.iloc[idx, df.columns.get_loc('miss_flag')] = []
                df.iloc[idx, df.columns.get_loc('body-MVE')] = []

                if args.modeltype == 'SMPLX':
                    df.iloc[idx, df.columns.get_loc('lhand-MPJPE')] = []
                    df.iloc[idx, df.columns.get_loc('rhand-MPJPE')] = []
                    df.iloc[idx, df.columns.get_loc('face-MPJPE')] = []
                    df.iloc[idx, df.columns.get_loc('lhand-MVE')] = []
                    df.iloc[idx, df.columns.get_loc('rhand-MVE')] = []
                    df.iloc[idx, df.columns.get_loc('face-MVE')] = []

                # collect all predicted joints. Might be smaller or larger than
                # gt
                for pNum in range(len(df.iloc[idx]['pred'])):
                    if df.iloc[idx].at['pred'][pNum]:
                        get_pred_joints(
                            predJoints, predVerts, df, idx, pNum, args)
                    else:
                        predJoints.append([])
                        predVerts.append([])

                if args.modeltype == 'SMPLX':
                    pel_aligned_error, pel_aligned_verts_error, lhand_joints_error, lhand_verts_error,\
                        rhand_joints_error, rhand_verts_error, face_kps_error, face_verts_error,\
                        falsePositives, miss_flag = v2v.compute_errors_matched_from_df_smplx_hf(args, df,
                                                                                                idx, predJoints, predVerts, verts_ind_dict)
                else:
                    pel_aligned_error, pel_aligned_verts_error, procrustes_error, procrustes_verts_error,\
                        falsePositives, procrustesParams, miss_flag = v2v.compute_errors_matched_from_df_smpl(args,
                                                                                                              df, idx,
                                                                                                              predJoints,
                                                                                                              verts_ind_dict,
                                                                                                              predVerts)

                # Update dataframes
                df.iloc[idx, df.columns.get_loc(
                    'falsePositives')] = falsePositives
                df.iloc[idx, df.columns.get_loc('misses')] = sum(miss_flag)

                for pNum in range(len(df.iloc[idx]['isValid'])):
                    df.iloc[idx, df.columns.get_loc(
                        'body-MPJPE')].append(pel_aligned_error[pNum])
                    df.iloc[idx, df.columns.get_loc(
                        'miss_flag')].append(miss_flag[pNum])
                    df.iloc[idx, df.columns.get_loc(
                        'body-MVE')].append(pel_aligned_verts_error[pNum])

                    if args.modeltype == 'SMPLX':
                        df.iloc[idx, df.columns.get_loc(
                            'lhand-MPJPE')].append(lhand_joints_error[pNum])
                        df.iloc[idx, df.columns.get_loc(
                            'rhand-MPJPE')].append(rhand_joints_error[pNum])
                        df.iloc[idx, df.columns.get_loc(
                            'face-MPJPE')].append(face_kps_error[pNum])
                        df.iloc[idx, df.columns.get_loc(
                            'lhand-MVE')].append(lhand_verts_error[pNum])
                        df.iloc[idx, df.columns.get_loc(
                            'rhand-MVE')].append(rhand_verts_error[pNum])
                        df.iloc[idx, df.columns.get_loc(
                            'face-MVE')].append(face_verts_error[pNum])
            # To reduce size, remove unused content from dataframe
            error_df_list.append(df)
        error_df = pandas.concat(error_df_list)
        error_df.to_pickle(
            os.path.join(
                args.result_savePath,
                args.baseline +
                '_df.pkl'))
    else:
        error_df = pandas.read_pickle(args.onlyComputeErrorLoadPath)
    logging.info('Calculating Average Error and Generating plots')
    compute_avg_error(args, error_df)

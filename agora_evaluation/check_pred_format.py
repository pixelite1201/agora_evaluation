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
import sys
import logging
import os

from glob import glob
import numpy as np
import pickle
import zipfile


logging.basicConfig(level=logging.DEBUG)

def assert_type(var, gt_type, key):
    if not isinstance(var, gt_type):
        raise TypeError('{} should be of type {}'.format(key, gt_type))

def assert_shape(var, gt_shape, key):
    if var.shape != gt_shape:
        raise ValueError('{} should be of shape {} but you are providing {}'.format(key, gt_shape, var.shape))

def assert_smpl_param_keys(params):
    assert 'transl' in params.keys()
    assert_type(params['transl'], np.ndarray, 'transl')
    assert 'betas' in params.keys()
    assert_type(params['betas'], np.ndarray, 'betas')
    assert 'global_orient' in params.keys()
    assert_type(params['global_orient'], np.ndarray, 'global_orient')
    assert 'body_pose' in params.keys()
    assert_type(params['body_pose'], np.ndarray, 'body_pose')

def assert_smpl_param_shape(params, pos2rot, num_betas):
    assert_shape(params['transl'], (1,3), 'transl')
    assert_shape(params['betas'], (1,num_betas),'betas')
    if pos2rot:
        assert_shape(params['global_orient'], (1,1,3), 'global_orient')
        assert_shape(params['body_pose'], (1,23,3), 'body_pose')
    else:
        assert_shape(params['global_orient'], (1,1,3,3), 'global_orient')
        assert_shape(params['body_pose'], (1,23,3,3), 'body_pose')


def assert_smplx_param_keys(params):
    assert 'transl' in params.keys()
    assert_type(params['transl'], np.ndarray, 'transl')
    assert 'betas' in params.keys()
    assert_type(params['betas'], np.ndarray, 'betas')
    assert 'global_orient' in params.keys()
    assert_type(params['global_orient'], np.ndarray, 'global_orient')
    assert 'body_pose' in params.keys()
    assert_type(params['body_pose'], np.ndarray, 'body_pose')
    assert 'left_hand_pose' in params.keys()
    assert_type(params['left_hand_pose'], np.ndarray, 'left_hand_pose')
    assert 'right_hand_pose' in params.keys()
    assert_type(params['right_hand_pose'], np.ndarray, 'right_hand_pose')
    assert 'leye_pose' in params.keys()
    assert_type(params['leye_pose'], np.ndarray, 'leye_pose')
    assert 'reye_pose' in params.keys()
    assert_type(params['reye_pose'], np.ndarray, 'reye_pose')
    assert 'jaw_pose' in params.keys()
    assert_type(params['jaw_pose'], np.ndarray, 'jaw_pose')
    assert 'expression' in params.keys()
    assert_type(params['expression'], np.ndarray, 'expression')


def assert_smplx_param_shape(params, pos2rot, num_betas):
    assert_shape(params['transl'], (1,3), 'transl')
    assert_shape(params['betas'], (1,num_betas),'betas')
    assert_shape(params['expression'], (1,10),'expression')
    if pos2rot:
        assert_shape(params['global_orient'], (1,3), 'global_orient')
        assert_shape(params['body_pose'], (1,63), 'body_pose')
        assert_shape(params['left_hand_pose'],(1,45),'left_hand_pose')
        assert_shape(params['right_hand_pose'],(1,45),'right_hand_pose')
        assert_shape(params['leye_pose'],(1,3),'leye_pose')
        assert_shape(params['reye_pose'],(1,3),'reye_pose')
        assert_shape(params['jaw_pose'],(1,3),'jaw_pose')
    else:
        assert_shape(params['global_orient'], (1,1,3,3), 'global_orient')
        assert_shape(params['body_pose'], (1,21,3,3), 'body_pose')
        assert_shape(params['left_hand_pose'],(1,15,3,3),'left_hand_pose')
        assert_shape(params['right_hand_pose'],(1,15,3,3),'right_hand_pose')
        assert_shape(params['leye_pose'],(1,1,3,3),'leye_pose')
        assert_shape(params['reye_pose'],(1,1,3,3),'reye_pose')
        assert_shape(params['jaw_pose'],(1,1,3,3),'jaw_pose')


def check_smpl(pred_file):
    pred_param = pickle.load(open(pred_file,'rb'),encoding='latin1')
    
    if 'allSmplJoints3d' in pred_param.keys() and 'verts' in pred_param.keys():
        joints3d = pred_param['allSmplJoints3d'].squeeze()
        verts3d = pred_param['verts'].squeeze()
        #Instance should be numpy array
        assert_type(verts3d, np.ndarray, 'verts')
        assert_type(joints3d, np.ndarray, 'allSmplJoints3d')
        #SMPL vertices shape should be (6890,3)
        assert_shape(verts3d, (6890,3), 'verts')
        if len(joints3d.shape)!=2 or joints3d.shape[1]!=3 or joints3d.shape[0]<24:
            raise ValueError('joints should be of shape (24,3) but you ar providing {}'.format(joints3d.shape))
        if joints3d.shape[0]>24:
            logging.warning(' Only first 24 3d joints will be used for body evaluation but you are providing {} joints'.format(joints3d.shape[0]))

    elif 'params' in pred_param.keys():
        #Default 10 betas will be used for SMPL adult and 11 for SMPL kid
        num_betas = 10
        #Optional parameter, default is adult
        if 'kid_flag' in pred_param.keys():
            kid_flag = pred_param['kid_flag']
            if kid_flag:
                num_betas=11
            if kid_flag not in [True, False]:
                raise KeyError('Either True or False should be provided in kid_flag. Found '.format(kid_flag))
            if kid_flag and 'params' in pred_param.keys() and pred_param['params']['betas'].shape[1] != 11:
                raise KeyError('For kid, 11 betas are used. Please check the ReadMe on Github')
                
        params = pred_param['params']
        #Check if all smpl params keys are present
        assert_smpl_param_keys(params)
        #Check shape of smpl params
        if 'pose2rot' in pred_param.keys():
            pose2rot=pred_param['pose2rot']
        else:
            pose2rot =True
        assert_smpl_param_shape(params,pose2rot, num_betas)
    else:
        raise KeyError('Either params or allSMPLJoints3d and verts needs to be provided in key. Please check the ReadMe for details and run the evaluation code on github')

    assert 'joints' in pred_param.keys()
    joints = pred_param['joints'].squeeze()
    assert_type(joints, np.ndarray,'joints')
    #Only first 24 joints will be used for matching
    if len(joints.shape)!=2 or joints.shape[1]<2 or joints.shape[0]<24:
        raise ValueError('joints should be of shape (24,2) but you are providing {}'.format(joints.shape))
    if joints.shape[0]>24:
        logging.warning(' Only first 24 joints will be used in matching but you are providing {} joints'.format(joints.shape[0]))

def check_smplx(pred_file):
    pred_param = pickle.load(open(pred_file,'rb'),encoding='latin1')
        
    if 'allSmplJoints3d' in pred_param.keys() and 'verts' in pred_param.keys():
        joints3d = pred_param['allSmplJoints3d'].squeeze()
        verts3d = pred_param['verts'].squeeze()
        #Instance should be numpy array
        assert_type(verts3d, np.ndarray, 'verts')
        assert_type(joints3d, np.ndarray, 'allSmplJoints3d')
        #SMPL-X vertices shape should be (10475,3)
        assert_shape(verts3d, (10475,3), 'verts')
        if len(joints3d.shape)!=2 or joints3d.shape[1]!=3 or joints3d.shape[0]<127:
            raise ValueError('joints should be of shape (127,3) but you ar providing {}'.format(joints3d.shape))
        if joints3d.shape[0]>127:
            logging.warning(' Only first 127 3d joints will be used for body,hands and face evaluation but you are providing {} joints'.format(joints3d.shape[0]))

    elif 'params' in pred_param.keys():
        num_betas = 10
        #Optional parameter, by default neutral
        if 'gender' in pred_param.keys():
            gender = pred_param['gender']
            if gender not in ['male', 'female', 'neutral']:
                raise KeyError('Gender {} is not correct. It should be either male, female or neutral'.format(gender))
        #Optional parameter, by default adult
        if 'kid_flag' in pred_param.keys():
            kid_flag = pred_param['kid_flag']
            if kid_flag not in [True, False]:
                raise KeyError('Either True or False should be provided in kid_flag. Found '.format(kid_flag))
            if kid_flag and 'params' in pred_param.keys() and pred_param['params']['betas'].shape[1] != 11:
                raise KeyError('For kid, 11 betas should be provided in betas. Please check the ReadMe on Github')
            if kid_flag and 'params' in pred_param.keys() and 'num_betas' in pred_param.keys() and pred_param['num_betas']!=11:
                raise ValueError('For kid, 11 betas should be provided in num_betas. Please check the ReadMe on Github')

        #Optional parameter, by default 10        
        if 'num_betas' in pred_param.keys():
            num_betas = pred_param['num_betas']
            if num_betas not in list(range(10,300)):
                raise KeyError('num_betas should be greater than 10 and less than 300. Found '.format(num_betas))
        
        params = pred_param['params']
        #Check if all smpl params keys are present
        assert_smplx_param_keys(params)
        #Check shape of smpl params
        if 'pose2rot' in pred_param.keys():
            pose2rot=pred_param['pose2rot']
        else:
            pose2rot=True
        assert_smplx_param_shape(params,pose2rot,num_betas)
        

    else:
        raise KeyError('Either params or allSMPLJoints3d and verts needs to be provided in key. Please check the ReadMe for details and run the evaluation code on github')

    assert 'joints' in pred_param.keys()
    joints = pred_param['joints']
    assert_type(pred_param['joints'],np.ndarray,'joints')
    #Only first 24 joints will be used for matching
    if len(joints.shape)!=2 or joints.shape[1]<2 or joints.shape[0]<24:
        raise ValueError('joints should be of shape (24,2) but you are providing {}'.format(joints.shape))
    if joints.shape[0]>24:
        logging.warning(' Only first 24 projected joints in joints key will be used in matching but you are providing {} joints'.format(joints.shape[0]))


def check_pred_file(*args):
    """Function to check the prediction file"""

    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--predZip', type=str,
                        default='')
    parser.add_argument('--extractZipFolder', type=str,
                        default='')
    parser.add_argument(
        '--modeltype',
        type=str,
        default='SMPLX',
        help='SMPL or SMPLX')

    args = parser.parse_args(*args)
    path_to_zip_file = args.predZip
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(args.extractZipFolder)

    all_files = glob(os.path.join(args.extractZipFolder, 'predictions', '*'))
    if len(all_files) == 0:
        raise EOFError('No files are present inside zip')
    if args.modeltype=='SMPL':
        for pred_file in all_files:
            logging.info('Reading file {}'.format(pred_file))
            check_smpl(pred_file)
    elif args.modeltype=='SMPLX':
        for pred_file in all_files:
            logging.info('Reading file {}'.format(pred_file))
            check_smplx(pred_file)
    else:
        raise KeyError('Only SMPL/SMPLX model type are supported')


if __name__=='__main__':
    check_pred_file(sys.argv[1:])
    logging.info('If you reach here then your zip folder is ready to submit')

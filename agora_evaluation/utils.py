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
import logging
import os

import numpy as np
import smplx


logging.basicConfig(level=logging.DEBUG)


def get_camYaw(args, constants):
    if args.scene3d and args.useCorrectMeanPoseRot:
        cam_yaw = 0
    elif args.useCorrectMeanPoseRot:
        cam_yaw = constants['camPitch']
    else:
        cam_yaw = 0
    return cam_yaw


def smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps OpenPose to SMPL

        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'

    '''
    if openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))


def convert_hmr23dpw(kp):
    new_kp = kp
    smplToLSP = [14, 12, 8, 7, 6, 9, 10, 11, 2, 1, 0, 3, 4, 5, 16, 15, 18, 17]

    new_kp = kp[smplToLSP, :]

    return new_kp


def SMPLX2HMR():
    return [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20, 12]


def convert_smpl2coco(kp):

    new_kp = kp
    # smplToLSP = [14, 12, 8, 7, 6, 9, 10, 11, 2,1,0, 4,5, 6, 16, 15, 18, 17]
    smplToLSP = [14, 15, 16, 17, 18, 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0]

    new_kp = kp[smplToLSP, :]

    return new_kp


def get_matching_dict(matching):
    matchDict = {}
    falsePositive_count = 0
    for match in matching:
        if not (match[1] == 'falsePositive') or match[0] == 'invalid':
            # tuple order (idx_openpose_pred, idx_gt_kps)
            matchDict[str(match[1])] = match[0]
        elif (match[1] == 'falsePositive'):
            falsePositive_count += 1
        else:
            continue  # simply ignore invalid ground truths
    return matchDict, falsePositive_count


def l2_error(j1, j2):
    return np.linalg.norm(j1 - j2, 2)


def load_model(args):

    if args.modeltype == 'SMPLX' and args.pose2rot:
        model_male = smplx.create(args.modelFolder, model_type='smplx',
                                  gender='male',
                                  ext='npz',
                                  num_betas=args.numBetas, use_pca=False)
        model_male_kid = smplx.create(args.modelFolder, model_type='smplx',
                                      gender='male',
                                      age='kid',
                                      kid_template_path=args.kid_template_path,
                                      ext='npz', use_pca=False)

        model_female = smplx.create(args.modelFolder, model_type='smplx',
                                    gender='female',
                                    ext='npz',
                                    num_betas=args.numBetas,
                                    use_pca=False)

        model_female_kid = smplx.create(
            args.modelFolder,
            model_type='smplx',
            gender='female',
            age='kid',
            kid_template_path=args.kid_template_path,
            ext='npz',
            use_pca=False)

        model_neutral = smplx.create(args.modelFolder, model_type='smplx',
                                     gender='neutral',
                                     ext='npz',
                                     num_betas=args.numBetas,
                                     use_pca=False)

        model_neutral_kid = smplx.create(
            args.modelFolder,
            model_type='smplx',
            gender='neutral',
            age='kid',
            kid_template_path=args.kid_template_path,
            ext='npz',
            use_pca=False)

    elif args.modeltype == 'SMPLX' and not args.pose2rot:
        # If params are in rotation matrix format then we need to use SMPLXLayer class
        model_male = smplx.build_layer(args.modelFolder, model_type='smplx',
                                  gender='male',
                                  ext='npz',
                                  num_betas=args.numBetas, use_pca=False)
        model_male_kid = smplx.build_layer(args.modelFolder, model_type='smplx',
                                      gender='male',
                                      age='kid',
                                      kid_template_path=args.kid_template_path,
                                      ext='npz', use_pca=False)

        model_female = smplx.build_layer(args.modelFolder, model_type='smplx',
                                    gender='female',
                                    ext='npz',
                                    num_betas=args.numBetas,
                                    use_pca=False)

        model_female_kid = smplx.build_layer(
            args.modelFolder,
            model_type='smplx',
            gender='female',
            age='kid',
            kid_template_path=args.kid_template_path,
            ext='npz',
            use_pca=False)

        model_neutral = smplx.build_layer(args.modelFolder, model_type='smplx',
                                     gender='neutral',
                                     ext='npz',
                                     num_betas=args.numBetas,
                                     use_pca=False)

        model_neutral_kid = smplx.build_layer(
            args.modelFolder,
            model_type='smplx',
            gender='neutral',
            age='kid',
            kid_template_path=args.kid_template_path,
            ext='npz',
            use_pca=False)

    elif args.modeltype == 'SMPL':
        model_male = smplx.create(args.modelFolder, model_type='smpl',
                                  gender='male',
                                  ext='npz')
        model_male_kid = smplx.create(args.modelFolder, model_type='smpl',
                                      gender='male', age='kid',
                                      kid_template_path=args.kid_template_path,
                                      ext='npz')
        model_female = smplx.create(args.modelFolder, model_type='smpl',
                                    gender='female',
                                    ext='npz')
        model_female_kid = smplx.create(
            args.modelFolder,
            model_type='smpl',
            gender='female',
            age='kid',
            kid_template_path=args.kid_template_path,
            ext='npz')
        model_neutral = smplx.create(args.modelFolder, model_type='smpl',
                                     gender='neutral',
                                     ext='npz')
        model_neutral_kid = smplx.create(
            args.modelFolder,
            model_type='smpl',
            gender='neutral',
            age='kid',
            kid_template_path=args.kid_template_path,
            ext='npz')
    else:
        raise ValueError('Provide correct modeltype smpl/smplx')
    return model_male, model_male_kid, model_female, model_female_kid, model_neutral, model_neutral_kid

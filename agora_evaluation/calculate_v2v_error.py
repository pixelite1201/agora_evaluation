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

import numpy as np

from . import utils


logging.basicConfig(level=logging.DEBUG)
SMPL_JOINTS = 24
SMPLX_JOINTS = 22


def get_spheres(points, color, radius):
    from psbody.mesh.sphere import Sphere

    points = points
    spheres = [
        Sphere(
            center=point,
            radius=radius).to_mesh(
            color=color) for point in points if all(point)]

    return spheres


def compute_similarity_transform(S1, S2, num_joints, verts=None):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        if verts is not None:
            verts = verts.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # Use only body joints for procrustes
    S1_p = S1[:, :num_joints]
    S2_p = S2[:, :num_joints]
    # 1. Remove mean.
    mu1 = S1_p.mean(axis=1, keepdims=True)
    mu2 = S2_p.mean(axis=1, keepdims=True)
    X1 = S1_p - mu1
    X2 = S2_p - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1 ** 2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if verts is not None:
        verts_hat = scale * R.dot(verts) + t
        if transposed:
            verts_hat = verts_hat.T

    if transposed:
        S1_hat = S1_hat.T

    procrustes_params = {'scale': scale,
                         'R': R,
                         'trans': t}

    if verts is not None:
        return S1_hat, verts_hat, procrustes_params
    else:
        return S1_hat, procrustes_params


def align_by_pelvis(joints, verts=None):

    left_id = 1
    right_id = 2

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.0
    if verts is not None:
        return verts - np.expand_dims(pelvis, axis=0)
    else:
        return joints - np.expand_dims(pelvis, axis=0)


def align_by_wrist(joints, verts=None):

    wrist_id = 0
    wrist = joints[wrist_id, :]

    if verts is not None:
        return verts - np.expand_dims(wrist, axis=0)
    else:
        return joints - np.expand_dims(wrist, axis=0)


def align_by_neck(joints, verts=None):

    neck_id = 0
    neck = joints[neck_id, :]

    if verts is not None:
        return verts - np.expand_dims(neck, axis=0)
    else:
        return joints - np.expand_dims(neck, axis=0)


def compute_errors_joints_verts(gt_verts, pred_verts, gt_joints,
                                pred_joints, miss, flag='body', debug=False):

    num_joints = gt_joints[0].shape[0]
    errors, errors_procrustes, procrustesParams = [], [], []
    errors_verts, errors_procrustes_verts = [], []

    for i, (gt3d, pred) in enumerate(zip(gt_joints, pred_joints)):
        # Get corresponding ground truth and predicted 3d joints and verts
        if miss[i] == 1:
            errors.append(0)
            errors_verts.append(0)
            errors_procrustes.append(0)
            errors_procrustes_verts.append(0)
            procrustesParams.append({'scale': 0,
                                     'R': 0,
                                     'trans': 0})
            continue
        gt3d = gt3d.reshape(-1, 3)
        gt3d_verts = gt_verts[i].reshape(-1, 3)
        pred3d_verts = pred_verts[i].reshape(-1, 3)
        # Root align.
        if flag == 'body':
            gt3d_verts = align_by_pelvis(gt3d, gt3d_verts)
            pred3d_verts = align_by_pelvis(pred, pred3d_verts)
            gt3d = align_by_pelvis(gt3d)
            pred3d = align_by_pelvis(pred)
        elif flag == 'face':
            gt_dict = {}
            pred_dict = {}
            gt3d_verts = align_by_neck(gt3d, gt3d_verts)
            pred3d_verts = align_by_neck(pred, pred3d_verts)
            gt3d = align_by_neck(gt3d)
            pred3d = align_by_neck(pred)
        elif flag == 'hand':
            gt3d_verts = align_by_wrist(gt3d, gt3d_verts)
            pred3d_verts = align_by_wrist(pred, pred3d_verts)
            gt3d = align_by_wrist(gt3d)
            pred3d = align_by_wrist(pred)
        else:
            raise KeyError(
                'Incorrect flag provided. Should be either body,face or hand')

        if debug:
            from psbody.mesh import MeshViewer, Mesh
            mv = MeshViewer()
            gt_spheres = get_spheres(
                gt3d * np.array([1, -1, -1]), [255, 0, 0], 0.01)
            pred_spheres = get_spheres(
                pred3d * np.array([1, -1, -1]), [0, 255, 0], 0.01)
            mesh1 = Mesh(v=gt3d_verts * np.array([1, -1, -1]))
            mesh2 = Mesh(v=pred3d_verts * np.array([1, -1, -1]))
            mv.set_static_meshes([mesh1, mesh2] + gt_spheres + pred_spheres)
            input('Enter something to continue')
        # Calculate joints and verts pelvis align error
        joint_error = np.sqrt(np.sum((gt3d - pred3d) ** 2, axis=1))
        verts_error = np.sqrt(np.sum((gt3d_verts - pred3d_verts) ** 2, axis=1))
        errors.append(np.mean(joint_error))
        errors_verts.append(np.mean(verts_error))

        # Get procrustes align error. # Not used anymore
        pred3d_sym, pred3d_verts_sym, procrustesParam = compute_similarity_transform(
            pred3d, gt3d, num_joints, pred3d_verts)

        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym) ** 2, axis=1))
        pa_verts_error = np.sqrt(
            np.sum(
                (gt3d_verts - pred3d_verts_sym) ** 2,
                axis=1))
        if debug:
            from psbody.mesh import MeshViewer, Mesh
            mv = MeshViewer()
            proc_gt_spheres = get_spheres(
                gt3d * np.array([1, -1, -1]), [255, 0, 0], 0.01)
            proc_pred_spheres = get_spheres(
                pred3d_sym * np.array([1, -1, -1]), [0, 255, 0], 0.01)
            mesh1 = Mesh(v=gt3d_verts * np.array([1, -1, -1]))
            mesh2 = Mesh(v=pred3d_verts_sym * np.array([1, -1, -1]))
            mv.set_static_meshes(
                [mesh1, mesh2] + proc_gt_spheres + proc_pred_spheres)
            input('Enter something to continue')

        errors_procrustes.append(np.mean(pa_error))
        errors_procrustes_verts.append(np.mean(pa_verts_error))
        procrustesParams.append(procrustesParam)

    return errors, errors_verts, errors_procrustes, errors_procrustes_verts, procrustesParams


def compute_errors_matched_from_df_smpl(
        args, df, idx, pred_j, verts_ind_dict, pred_v=None):

    body_verts_ind = verts_ind_dict['body-smpl']
    gt_joints_3d = df.iloc[idx]['gt_joints_3d']
    gt_verts_3d = df.iloc[idx]['gt_verts']
    matching = df.iloc[idx].at['matching']
    njoints = SMPL_JOINTS

    gt3d_verts, pred_verts, gt3d_joints, pred_joints = [], [], [], []

    matchDict, falsePositive_count = utils.get_matching_dict(matching)
    gtIdxs = np.arange(len(gt_joints_3d))
    miss_flag = []
    for gtIdx in gtIdxs:
        gt3d_verts.append(gt_verts_3d[gtIdx][body_verts_ind, :])
        gt3d_joints.append(gt_joints_3d[gtIdx][:njoints, :])
        if matchDict[str(gtIdx)] == 'miss' or matchDict[str(
                gtIdx)] == 'invalid':
            miss_flag.append(1)
            pred_verts.append([])
            pred_joints.append([])
        else:
            miss_flag.append(0)
            pred_joints.append(pred_j[matchDict[str(gtIdx)]][:njoints, :])
            pred_verts.append(pred_v[matchDict[str(gtIdx)]][body_verts_ind, :])

    errors_pelvis, errors_pelvis_verts, errors_procrustes, errors_procrustes_verts, procrustesParams =\
        compute_errors_joints_verts(gt3d_verts, pred_verts, gt3d_joints, pred_joints, miss_flag, debug=False)
    return errors_pelvis, errors_pelvis_verts, errors_procrustes, errors_procrustes_verts,\
        falsePositive_count, procrustesParams, miss_flag


def compute_errors_matched_from_df_smplx_hf(
        args, df, idx, pred_j, pred_v, verts_ind_dict):
    # Get vertex indices for different body parts
    lh_verts_ind = np.array(verts_ind_dict['left_hand'])
    rh_verts_ind = np.array(verts_ind_dict['right_hand'])
    body_verts_ind = verts_ind_dict['body-smplx']
    face_verts_ind = verts_ind_dict['face']

    # Get gt joints and vertices
    gt_joints_3d = df.iloc[idx]['gt_joints_3d']
    gt_verts_3d = df.iloc[idx]['gt_verts']
    matching = df.iloc[idx].at['matching']

    gt3d_verts, pred_verts, gt3d_joints, pred_joints = [], [], [], []
    gt3d_joints_l, gt3d_joints_r, pred_joints_l,\
        pred_joints_r, gt3d_joints_f, pred_joints_f = [], [], [], [], [], []
    gt3d_verts_l, gt3d_verts_r, pred_verts_l,\
        pred_verts_r, gt3d_verts_f, pred_verts_f = [], [], [], [], [], []

    matchDict, falsePositive_count = utils.get_matching_dict(matching)

    gtIdxs = np.arange(len(gt_joints_3d))
    miss_flag, gt_verts_all, pred_verts_all,\
        gt_joints_all, pred_joints_all = [], [], [], [], []

    for gtIdx in gtIdxs:
        # Add gt verts
        gt_verts_all.append(gt_verts_3d[gtIdx])
        gt_joints_all.append(gt_joints_3d[gtIdx])
        gt3d_verts.append(gt_verts_3d[gtIdx][body_verts_ind, :])
        gt3d_verts_l.append(gt_verts_3d[gtIdx][lh_verts_ind, :])
        gt3d_verts_r.append(gt_verts_3d[gtIdx][rh_verts_ind, :])
        gt3d_verts_f.append(gt_verts_3d[gtIdx][face_verts_ind, :])

        # Add gt joints, 20-left wrist, 21-right wrist, 12-neck,
        # 0-22 (body jonits), 25-40 (lhand joints), 40-55, (rhand joints),56+
        # (face joints)
        leftHand = np.concatenate([gt_joints_3d[gtIdx][20, :].reshape((1, 3)),
                                   gt_joints_3d[gtIdx][25:40, :]], axis=0)
        rightHand = np.concatenate([gt_joints_3d[gtIdx][21, :].reshape((1, 3)),
                                    gt_joints_3d[gtIdx][40:55, :]], axis=0)
        gt_face = np.concatenate([gt_joints_3d[gtIdx][12, :].reshape((1, 3)),
                                  gt_joints_3d[gtIdx][76:127, :]], axis=0)
        gt3d_joints.append(gt_joints_3d[gtIdx][:22, :])
        gt3d_joints_l.append(leftHand)
        gt3d_joints_r.append(rightHand)
        gt3d_joints_f.append(gt_face)

        if matchDict[str(gtIdx)] == 'miss' or matchDict[str(
                gtIdx)] == 'invalid':
            miss_flag.append(1)
            pred_verts_all.append([])
            pred_joints_all.append([])
            pred_verts.append([])
            pred_joints.append([])
            pred_joints_l.append([])
            pred_joints_r.append([])
            pred_joints_f.append([])
            pred_verts_l.append([])
            pred_verts_r.append([])
            pred_verts_f.append([])
        else:
            miss_flag.append(0)
            # Add pred joints
            pred_joints_id = pred_j[matchDict[str(gtIdx)]]
            pred_joints.append(pred_joints_id[:22, :])
            pred_joints_all.append(pred_joints_id[:127, :])
            pred_left = np.concatenate(
                [pred_joints_id[20, :].reshape((1, 3)), pred_joints_id[25:40, :]], axis=0)
            pred_right = np.concatenate(
                [pred_joints_id[21, :].reshape((1, 3)), pred_joints_id[40:55, :]], axis=0)
            pred_face = np.concatenate([pred_joints_id[12, :].reshape(
                (1, 3)), pred_joints_id[76:127, :]], axis=0)
            pred_joints_l.append(pred_left)
            pred_joints_r.append(pred_right)
            pred_joints_f.append(pred_face)
            # Add pred verts
            all_pred_verts = pred_v[matchDict[str(gtIdx)]]
            pred_verts_all.append(all_pred_verts)
            pred_verts.append(all_pred_verts[body_verts_ind, :])
            pred_verts_l.append(all_pred_verts[lh_verts_ind, :])
            pred_verts_r.append(all_pred_verts[rh_verts_ind, :])
            pred_verts_f.append(all_pred_verts[face_verts_ind, :])

    errors_pelvis, errors_pelvis_verts, errors_procrustes, errors_procrustes_verts, procrustesParams =\
        compute_errors_joints_verts(gt3d_verts, pred_verts, gt3d_joints, pred_joints, miss_flag, debug=False)

    errors_pelvis_l, errors_pelvis_verts_l, errors_procrustes_l, errors_procrustes_verts_l, procrustesParams_l =\
        compute_errors_joints_verts(gt3d_verts_l, pred_verts_l, gt3d_joints_l, pred_joints_l, miss_flag, flag='hand', debug=False)

    errors_pelvis_r, errors_pelvis_verts_r, errors_procrustes_r, errors_procrustes_verts_r, procrustesParams_r =\
        compute_errors_joints_verts(gt3d_verts_r, pred_verts_r, gt3d_joints_r, pred_joints_r, miss_flag, flag='hand', debug=False)

    errors_pelvis_f, errors_pelvis_verts_f, errors_procrustes_f, errors_procrustes_verts_f, procrustesParams_f = \
        compute_errors_joints_verts(gt3d_verts_f, pred_verts_f, gt3d_joints_f, pred_joints_f, miss_flag, flag='face', debug=False)

    return errors_pelvis, errors_pelvis_verts, errors_pelvis_l, errors_pelvis_verts_l,\
        errors_pelvis_r, errors_pelvis_verts_r, errors_pelvis_f,\
        errors_pelvis_verts_f, falsePositive_count, miss_flag

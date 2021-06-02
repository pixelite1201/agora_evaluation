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
import math
import os
import pickle
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

scalar = 1e3
occ_bins = [-1, 10, 20, 30, 40, 50, 60, 70, 80]
ori_bins = [-1, 45, 90, 135, 180, 225, 270, 315, 360]
x_bins = [-1, 192, 384, 576, 768, 960, 1152, 1344, 1536, 1728, 1920]
x_bins_1280 = [-1, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640]

logging.basicConfig(level=logging.DEBUG)


def compute_prf1(count, miss, num_fp):
    if count == 0:
        return 0, 0, 0
    all_tp = count - miss
    all_fp = len(num_fp)
    all_fn = miss
    all_f1_score = round(all_tp / (all_tp + 0.5 * (all_fp + all_fn)), 2)
    all_recall = round(all_tp / (all_tp + all_fn), 2)
    all_precision = round(all_tp / (all_tp + all_fp), 2)
    return all_precision, all_recall, all_f1_score


def create_normalized_df(df, flag='occ'):

    if flag == 'occ':
        df['occ_bins'] = pd.cut(np.array(df.iloc[:, 0]), occ_bins)
        df_without_miss = df.loc[df['Miss'] == 0]
        df_without_miss = df_without_miss.groupby(['occ_bins']).agg(['mean'])
        df_miss_percentage = df.groupby(['occ_bins']).agg(['mean'])
    if flag == 'ori':
        df['ori_bins'] = pd.cut(np.array(df.iloc[:, 0]), ori_bins)
        df_without_miss = df.loc[df['Miss'] == 0]
        df_without_miss = df_without_miss.groupby(['ori_bins']).agg(['mean'])
        df_miss_percentage = df.groupby(['ori_bins']).agg(['mean'])

    df_without_miss.reset_index(inplace=True)
    df_without_miss.columns = df_without_miss.columns.droplevel(1)

    df_miss_percentage.reset_index(inplace=True)
    df_miss_percentage.columns = df_miss_percentage.columns.droplevel(1)
    columns_to_divide = ['body-MPJPE', 'body-MVE']
    df_without_miss[columns_to_divide] = (
        df_without_miss[columns_to_divide].div(
            1 - df_miss_percentage['Miss'], axis=0))
    return df_without_miss


def create_unnormalized_df(df, imgWidth, flag='occ'):
    df_new = df.loc[df['Miss'] == 0].copy()

    if flag == 'occ':
        df_new['occ_bins'] = pd.cut(np.array(df_new.iloc[:, 0]), occ_bins)
        df_new = df_new.groupby(['occ_bins']).agg(['mean'])

    if flag == 'ori':
        df_new['ori_bins'] = pd.cut(np.array(df_new.iloc[:, 0]), ori_bins)
        df_new = df_new.groupby(['ori_bins']).agg(['mean'])

    if flag == 'X' and imgWidth==3840:
        df_new['x_bins'] = pd.cut(np.array(df_new.iloc[:, 0]), x_bins)
        df_new = df_new.groupby(['x_bins']).agg(['mean'])

    if flag == 'X' and imgWidth==1280:
        df_new['x_bins'] = pd.cut(np.array(df_new.iloc[:, 0]), x_bins_1280)
        df_new = df_new.groupby(['x_bins']).agg(['mean'])

    df_new.reset_index(inplace=True)
    df_new.columns = df_new.columns.droplevel(1)
    return df_new


def plot_x_error(df, x, y, xlabel, ylabel, title, outfile):

    sns.set(style="whitegrid")

    ax = sns.catplot(x=x, y=y, data=df, kind='bar')
    ax.fig.set_size_inches(20, 10)

    axes = ax.axes.flatten()
    axes[0].set_xlabel(xlabel=xlabel, fontsize=20)
    axes[0].set_ylabel(ylabel=ylabel, fontsize=20)
    ax.set_xticklabels(['0-10',
                        '10-20',
                        '20-30',
                        '30-40',
                        '40-50',
                        '50-60',
                        '60-70',
                        '70-80',
                        '80-90',
                        '90-100'],
                       rotation='horizontal',
                       fontsize=20)
    ax.set_yticklabels(
        rotation='horizontal', fontsize=20)
    # ax.set_yticklabels(['20', '40', '60', '70', '80', '100', '120', '140','160'],
    #                    rotation='horizontal',fontsize=30)
    axes[0].set_title(title, fontsize=20)
    axes[0].legend(loc='upper left', fontsize=20)
    plt.savefig(outfile)


def plot_occ_error(df, x, y, xlabel, ylabel, title, outfile):

    sns.set(style="whitegrid")

    ax = sns.catplot(x=x, y=y, data=df, kind='bar')
    ax.fig.set_size_inches(20, 10)

    axes = ax.axes.flatten()
    axes[0].set_xlabel(xlabel=xlabel, fontsize=20)
    axes[0].set_ylabel(ylabel=ylabel, fontsize=20)

    ax.set_xticklabels(['0-10',
                        '10-20',
                        '20-30',
                        '30-40',
                        '40-50',
                        '50-60',
                        '60-70',
                        '70-80',
                        '80-90',
                        '90-100'],
                       rotation='horizontal',
                       fontsize=20)
    ax.set_yticklabels(
        rotation='horizontal', fontsize=20)
    axes[0].set_title(title, fontsize=20)
    axes[0].legend(loc='upper left', fontsize=20)
    plt.savefig(outfile)


def plot_occ_error(df, x, y, xlabel, ylabel, title, outfile):

    sns.set(style="whitegrid")

    ax = sns.catplot(x=x, y=y, data=df, kind='bar')
    ax.fig.set_size_inches(20, 10)

    axes = ax.axes.flatten()
    axes[0].set_xlabel(xlabel=xlabel, fontsize=20)
    axes[0].set_ylabel(ylabel=ylabel, fontsize=20)

    ax.set_xticklabels(['0-10',
                        '10-20',
                        '20-30',
                        '30-40',
                        '40-50',
                        '50-60',
                        '60-70',
                        '70-80'],
                       rotation='horizontal',
                       fontsize=20)
    ax.set_yticklabels(
        rotation='horizontal', fontsize=20)
    axes[0].set_title(title, fontsize=20)
    axes[0].legend(loc='upper left', fontsize=20)
    plt.savefig(outfile)


def plot_ori_error(df, x, y, xlabel, ylabel, title, outfile):
    sns.set(style="whitegrid")

    ax = sns.catplot(x=x, y=y, data=df, kind='bar')
    ax.fig.set_size_inches(20, 10)

    axes = ax.axes.flatten()
    axes[0].set_xlabel(xlabel=xlabel, fontsize=20)
    axes[0].set_ylabel(ylabel=ylabel, fontsize=20)

    ax.set_xticklabels(['0$\\degree$-45$\\degree$',
                        '45$\\degree$-90$\\degree$',
                        '90$\\degree$-135$\\degree$',
                        '135$\\degree$-180$\\degree$',
                        '180$\\degree$-225$\\degree$',
                        '225$\\degree$-270$\\degree$',
                        '270$\\degree$-315$\\degree$',
                        '315$\\degree$-360$\\degree$'],
                       rotation=45,
                       fontsize=20)

    axes[0].set_title(title, fontsize=20)
    axes[0].legend(loc='upper left', fontsize=18)

    plt.savefig(outfile)


def compute_avg_error(args, df):
    total_miss_count = 0
    total_count = 0
    kid_miss_count = 0
    kid_total_count = 0
    lhand_joints_err = []
    rhand_joints_err = []
    lhand_verts_err = []
    rhand_verts_err = []
    face_joints_err = []
    face_verts_err = []
    kid_lhand_joints_err = []
    kid_rhand_joints_err = []
    kid_lhand_verts_err = []
    kid_rhand_verts_err = []
    kid_face_joints_err = []
    kid_face_verts_err = []
    fp = []
    occ_error = []
    ori_error = []
    x_error = []
    for i in range(len(df)):
        # If prediction is available for the image
        imgPath = df.iloc[i, df.columns.get_loc('imgPath')]
        if args.onlybfh and ('bfh' not in imgPath and 'smplx' not in imgPath):
            continue
        if df.iloc[i, df.columns.get_loc('pred_path')]:
            for j, valid in enumerate(
                    df.iloc[i, df.columns.get_loc('isValid')]):
                # If valid i.e. occlusion >0 and occlusion<90 percent
                if valid:
                    # Get occlusion and orientation value for plotting
                    occ = df.iloc[i, df.columns.get_loc('occlusion')][j]
                    camYaw = df.iloc[i]['camYaw']
                    if 'YawLocal' in df and not math.isnan(
                            df.iloc[i]['YawLocal'][j]):
                        orient = (df.iloc[i, df.columns.get_loc(
                            'YawLocal')][j] - 90 + 22.5 - camYaw) % 360
                        if orient < 0:
                            orient += 360

                    elif 'Yaw' in df and not math.isnan(df.iloc[i]['Yaw'][j]):
                        orient = (df.iloc[i, df.columns.get_loc(
                            'Yaw')][j] - 90 + 22.5 - camYaw) % 360
                        if orient < 0:
                            orient += 360
                    else:
                        logging.debug(
                            'either Yaw or YawLocal should be in the df')
                        exit(-1)

                    kid = df.iloc[i, df.columns.get_loc('kid')][j]
                    miss = df.iloc[i, df.columns.get_loc('miss_flag')][j]
                    x_location = abs(df.iloc[i, df.columns.get_loc(
                        'gt_joints_2d')][j][0, 0] - (args.imgWidth / 2))

                    total_count += 1
                    if kid:
                        kid_total_count += 1
                        # Because kid are only in bfh data so hand and face can
                        # be evaluted

                    # Just store the miss count to divide by it in the end.
                    if miss == 1:
                        total_miss_count += 1
                        if kid:
                            kid_miss_count += 1

                    # Get error, will be empty for misses
                    err_joints_body = df.iloc[i,
                                              df.columns.get_loc('body-MPJPE')][j]
                    if args.modeltype == 'SMPLX':
                        lhand_joints_err.append(
                            scalar * df.iloc[i, df.columns.get_loc('lhand-MPJPE')][j])
                        rhand_joints_err.append(
                            scalar * df.iloc[i, df.columns.get_loc('rhand-MPJPE')][j])
                        face_joints_err.append(
                            scalar * df.iloc[i, df.columns.get_loc('face-MPJPE')][j])
                        lhand_verts_err.append(
                            scalar * df.iloc[i, df.columns.get_loc('lhand-MVE')][j])
                        rhand_verts_err.append(
                            scalar * df.iloc[i, df.columns.get_loc('rhand-MVE')][j])
                        face_verts_err.append(
                            scalar * df.iloc[i, df.columns.get_loc('face-MVE')][j])
                        if kid:

                            kid_lhand_joints_err.append(
                                scalar * df.iloc[i, df.columns.get_loc('lhand-MPJPE')][j])
                            kid_rhand_joints_err.append(
                                scalar * df.iloc[i, df.columns.get_loc('rhand-MPJPE')][j])
                            kid_face_joints_err.append(
                                scalar * df.iloc[i, df.columns.get_loc('face-MPJPE')][j])
                            kid_lhand_verts_err.append(
                                scalar * df.iloc[i, df.columns.get_loc('lhand-MVE')][j])
                            kid_rhand_verts_err.append(
                                scalar * df.iloc[i, df.columns.get_loc('rhand-MVE')][j])
                            kid_face_verts_err.append(
                                scalar * df.iloc[i, df.columns.get_loc('face-MVE')][j])

                    err_verts_body = df.iloc[i,
                                             df.columns.get_loc('body-MVE')][j]

                    occ_error.append([occ, scalar * err_joints_body,
                                      scalar * err_verts_body, kid, miss])

                    ori_error.append([orient, scalar * err_joints_body,
                                      scalar * err_verts_body, kid, miss])

                    x_error.append([x_location, scalar * err_joints_body,
                                    scalar * err_verts_body, kid, miss])

            for j in range(df.iloc[i, df.columns.get_loc('falsePositives')]):
                fp.append(df.iloc[i, df.columns.get_loc('pred_path')])
        else:
            for j, valid in enumerate(
                    df.iloc[i, df.columns.get_loc('isValid')]):
                if valid:
                    miss = 1
                    total_count += 1
                    total_miss_count += 1
                    kid = df.iloc[i, df.columns.get_loc('kid')][j]
                    occ = df.iloc[i, df.columns.get_loc('occlusion')][j]
                    camYaw = df.iloc[i]['camYaw']
                    if 'YawLocal' in df and not math.isnan(
                            df.iloc[i]['YawLocal'][j]):
                        orient = (df.iloc[i, df.columns.get_loc(
                            'YawLocal')][j] - 90 + 22.5 - camYaw) % 360
                        if orient < 0:
                            orient += 360

                    elif 'Yaw' in df and not math.isnan(df.iloc[i]['Yaw'][j]):
                        orient = (df.iloc[i, df.columns.get_loc(
                            'Yaw')][j] - 90 + 22.5 - camYaw) % 360
                        if orient < 0:
                            orient += 360
                    else:
                        logging.debug(
                            'either Yaw or YawLocal should be in the df')
                        exit(-1)

                    x_location = abs(df.iloc[i, df.columns.get_loc(
                        'gt_joints_2d')][j][0, 0] - (args.imgWidth / 2))
                    if kid:
                        kid_total_count += 1
                        kid_miss_count += 1
                    occ_error.append([occ, 0, 0, kid, miss])
                    ori_error.append([orient, 0, 0, kid, miss])
                    x_error.append([x_location, 0, 0, kid, miss])

    precision, recall, f1 = compute_prf1(total_count, total_miss_count, fp)
    if kid_total_count != 0:
        kid_precision, kid_recall, kid_f1 = compute_prf1(
            kid_total_count, kid_miss_count, fp)
    else:
        kid_precision, kid_recall, kid_f1 = 'nan', 'nan', 'nan'

    occ_df = pd.DataFrame(
        occ_error,
        columns=[
            'occ (%)',
            'body-MPJPE',
            'body-MVE',
            'Kid',
            'Miss'])
    ori_df = pd.DataFrame(
        ori_error,
        columns=[
            'orient',
            'body-MPJPE',
            'body-MVE',
            'Kid',
            'Miss'])
    occ_df = pd.DataFrame(
        occ_error,
        columns=[
            'occ (%)',
            'body-MPJPE',
            'body-MVE',
            'Kid',
            'Miss'])
    ori_df = pd.DataFrame(
        ori_error,
        columns=[
            'orient',
            'body-MPJPE',
            'body-MVE',
            'Kid',
            'Miss'])
    x_df = pd.DataFrame(
        x_error,
        columns=[
            'X',
            'body-MPJPE',
            'body-MVE',
            'Kid',
            'Miss'])
    mean_occ_df = create_unnormalized_df(occ_df, args.imgWidth, flag='occ')
    mean_x_df = create_unnormalized_df(x_df, args.imgWidth, flag='X')
    mean_ori_df = create_unnormalized_df(ori_df, args.imgWidth, flag='ori')
    norm_occ_df = create_normalized_df(occ_df, flag='occ')
    norm_ori_df = create_normalized_df(ori_df, flag='ori')
    occ_df.to_pickle(
        os.path.join(
            args.result_savePath,
            args.baseline +
            '_miss.npy'))
    mean_x_df.to_pickle(
        os.path.join(
            args.result_savePath,
            args.baseline +
            '_mean_x_df.npy'))
    mean_occ_df.to_pickle(
        os.path.join(
            args.result_savePath,
            args.baseline +
            '_mean_occ_df.npy'))
    mean_ori_df.to_pickle(
        os.path.join(
            args.result_savePath,
            args.baseline +
            '_mean_ori_df.npy'))
    norm_occ_df.to_pickle(
        os.path.join(
            args.result_savePath,
            args.baseline +
            '_norm_occ_df.npy'))
    norm_ori_df.to_pickle(
        os.path.join(
            args.result_savePath,
            args.baseline +
            '_norm_ori_df.npy'))

    plot_occ_error(
        mean_occ_df,
        'occ (%)',
        'body-MPJPE',
        'occlusion (%)',
        'body-MPJPE',
        args.baseline,
        os.path.join(
            args.result_savePath,
            args.baseline +
            '_occlusion_mpjpe.png'))

    plot_ori_error(
        mean_ori_df,
        'orient',
        'body-MPJPE',
        'orientation',
        'body-MPJPE',
        args.baseline,
        os.path.join(
            args.result_savePath,
            args.baseline +
            '_orientation_mpjpe.png'))

    plot_x_error(
        mean_x_df,
        'X',
        'body-MPJPE',
        'Img (X%)',
        'body-MPJPE',
        args.baseline,
        os.path.join(
            args.result_savePath,
            args.baseline +
            '_xlocation_mpjpe.png'))

    plot_occ_error(
        norm_occ_df,
        'occ (%)',
        'body-MPJPE',
        'occlusion (%)',
        'body-NMJE',
        args.baseline,
        os.path.join(
            args.result_savePath,
            args.baseline +
            '_occlusion_nmje.png'))

    plot_ori_error(
        norm_ori_df,
        'orient',
        'body-MPJPE',
        'orientation',
        'body-NMJE',
        args.baseline,
        os.path.join(
            args.result_savePath,
            args.baseline +
            '_orientation_nmje.png'))

    occ_df_without_miss = occ_df[occ_df['Miss'] == 0]
    error_dict = {}
    error_dict['precision'] = precision
    error_dict['recall'] = recall
    error_dict['f1'] = f1
    error_dict['kid_precision'] = kid_precision
    error_dict['kid_recall'] = kid_recall
    error_dict['kid_f1'] = kid_f1
    error_dict['body-MPJPE'] = round(
        occ_df_without_miss['body-MPJPE'].mean(), 1)
    error_dict['kid-body-MPJPE'] = round(
        occ_df_without_miss[occ_df_without_miss['Kid'] == 1]['body-MPJPE'].mean(), 1)
    error_dict['body-NMJE'] = round(error_dict['body-MPJPE'] / (f1), 1)
    if kid_total_count != 0:
        error_dict['kid-body-NMJE'] = round(
            error_dict['kid-body-MPJPE'] / (kid_f1), 1)
    else:
        error_dict['kid-body-NMJE'] = 'nan'


    plot_occ_error(
        mean_occ_df,
        'occ (%)',
        'body-MVE',
        'occlusion (%)',
        'body-MVE',
        args.baseline,
        os.path.join(
            args.result_savePath,
            args.baseline +
            '_occlusion_mve.png'))
    plot_ori_error(
        mean_ori_df,
        'orient',
        'body-MVE',
        'orientation',
        'body-MVE',
        args.baseline,
        os.path.join(
            args.result_savePath,
            args.baseline +
            '_orientation_mve.png'))

    plot_occ_error(
        norm_occ_df,
        'occ (%)',
        'body-MVE',
        'occlusion (%)',
        'body-NMVE',
        args.baseline,
        os.path.join(
            args.result_savePath,
            args.baseline +
            '_occlusion_nmve.png'))

    plot_ori_error(
        norm_ori_df,
        'orient',
        'body-MVE',
        'orientation',
        'body-NMVE',
        args.baseline,
        os.path.join(
            args.result_savePath,
            args.baseline +
            '_orientation_nmve.png'))

    error_dict['body-MVE'] = round(occ_df_without_miss['body-MVE'].mean(), 1)
    error_dict['kid-body-MVE'] = round(
        occ_df_without_miss[occ_df_without_miss['Kid'] == 1]['body-MVE'].mean(), 1)
    error_dict['body-NMVE'] = round(error_dict['body-MVE'] / (f1), 1)
    if kid_total_count != 0:
        error_dict['kid-body-NMVE'] = round(
            error_dict['kid-body-MVE'] / (kid_f1), 1)
    else:
        error_dict['kid-body-NMVE'] = 'nan'

    if args.modeltype == 'SMPLX':
        error_dict['lhand-MPJPE'] = round(
            np.mean(np.array(lhand_joints_err)), 1)
        error_dict['rhand-MPJPE'] = round(
            np.mean(np.array(rhand_joints_err)), 1)
        error_dict['face-MPJPE'] = round(np.mean(np.array(face_joints_err)), 1)
        error_dict['lhand-MVE'] = round(np.mean(np.array(lhand_verts_err)), 1)
        error_dict['rhand-MVE'] = round(np.mean(np.array(rhand_verts_err)), 1)
        error_dict['face-MVE'] = round(np.mean(np.array(face_verts_err)), 1)
        error_dict['fullbody-MPJPE'] = round(error_dict['body-MPJPE'] + (1 / 3) * error_dict['lhand-MPJPE'] + (
            1 / 3) * error_dict['rhand-MPJPE'] + (1 / 3) * error_dict['face-MPJPE'], 1)
        error_dict['fullbody-MVE'] = round(error_dict['body-MVE'] + (1 / 3) * error_dict['lhand-MVE'] + (
            1 / 3) * error_dict['rhand-MVE'] + (1 / 3) * error_dict['face-MVE'], 1)
        error_dict['fullbody-NMJE'] = round(
            error_dict['fullbody-MPJPE'] / (f1), 1)
        error_dict['fullbody-NMVE'] = round(
            error_dict['fullbody-MVE'] / (f1), 1)

        # All kids are in bfh
        error_dict['kid-lhand-MPJPE'] = round(
            np.mean(np.array(kid_lhand_joints_err)), 1)
        error_dict['kid-rhand-MPJPE'] = round(
            np.mean(np.array(kid_rhand_joints_err)), 1)
        error_dict['kid-face-MPJPE'] = round(
            np.mean(np.array(kid_face_joints_err)), 1)
        error_dict['kid-lhand-MVE'] = round(
            np.mean(np.array(kid_lhand_verts_err)), 1)
        error_dict['kid-rhand-MVE'] = round(
            np.mean(np.array(kid_rhand_verts_err)), 1)
        error_dict['kid-face-MVE'] = round(
            np.mean(np.array(kid_face_verts_err)), 1)
        error_dict['kid-fullbody-MPJPE'] = round(error_dict['kid-body-MPJPE'] + (1 / 3) * error_dict['kid-lhand-MPJPE'] + (
            1 / 3) * error_dict['kid-rhand-MPJPE'] + (1 / 3) * error_dict['kid-face-MPJPE'], 1)
        error_dict['kid-fullbody-MVE'] = round(error_dict['kid-body-MVE'] + (1 / 3) * error_dict['kid-lhand-MVE'] + (
            1 / 3) * error_dict['kid-rhand-MVE'] + (1 / 3) * error_dict['kid-face-MVE'], 1)
        error_dict['kid-fullbody-NMJE'] = round(
            error_dict['kid-fullbody-MPJPE'] / (f1), 1)
        error_dict['kid-fullbody-NMVE'] = round(
            error_dict['kid-fullbody-MVE'] / (f1), 1)

    logging.info(error_dict)

    pickle.dump(
        error_dict,
        open(
            os.path.join(
                args.result_savePath,
                args.baseline +
                '_result.pkl'),
            'wb'))

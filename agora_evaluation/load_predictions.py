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
from collections import defaultdict
import re
import pickle
from glob import glob


logging.basicConfig(level=logging.DEBUG)


def make_df_complete_pred(df, pred_dict):
    df.insert(column='pred_path', loc=len(df.columns), value=len(df) * [''])
    df.insert(column='pred', loc=len(df.columns), value=len(df) * [''])

    for idx in (range(len(df))):
        for jdx in range(len(df.iloc[idx]['isValid'])):
            # necessary to make each list unique
            if jdx == 0:
                df.iloc[idx, df.columns.get_loc('pred_path')] = []

        imgName = df.iloc[idx]['imgPath']

        for jdx in range(len(pred_dict[imgName])):
            df.iloc[idx].at["pred_path"].append(
                pred_dict[imgName][jdx])  # not same order)
    return df


def test_img_predMatch(args, df):
    # Todo Remove for github version
    for i in range(len(df)):
        if len(df.iloc[i]['pred_path']) == 0:
            logging.debug(
                'no prediction for frame ' +
                str(i) +
                'at ' +
                df.iloc[i]['imgPath'])
        else:
            df.iloc[i]['pred_path'].sort()
            predPath = df.iloc[i]['pred_path'][0].split('/')[-1]
            predictionName = re.sub('_personId_\\d*.pkl', '', predPath)
            imgName = df.iloc[i]['imgPath'].split('.')[0]
            #logging.debug("Found for imgname: {} corresponding prediction filename: {}".format(imgName, predictionName))
            assert imgName == predictionName


def load(args, df):
    # build up prediction dictionary (mapping from image name to predictions)
    predictions_path = glob(os.path.join(args.pred_path, '*.pkl'))
    pred_dict = defaultdict(list)
    if len(predictions_path) == 0:
        raise FileNotFoundError('No predictions!')
    # get idx to path mapping
    for pred_path in predictions_path:
        predictionName = pred_path.split('/')[-1]
        if predictionName == 'conf.yaml':
            continue
        orig_imgName = '_'.join(predictionName.split('_')[0:-2]) + '.png'
        pred_dict[orig_imgName].append(pred_path)

    logging.info('reading dataframe')
    df = make_df_complete_pred(df, pred_dict)
    test_img_predMatch(args, df)

    for idx in (range(len(df))):
        for jdx in range(len(df.iloc[idx]['pred_path'])):
            if jdx == 0:
                df.iloc[idx, df.columns.get_loc('pred')] = []
            # get prediction
            with open(df.iloc[idx]['pred_path'][jdx], 'rb') as infile:
                pred = pickle.load(infile, encoding='latin1')
            df.iloc[idx].at['pred'].append(pred)
    return df

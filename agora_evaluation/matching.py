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
from itertools import product
import logging
import os

import cv2
import numpy as np

from .utils import l2_error, smpl_to_openpose, SMPLX2HMR


logging.basicConfig(level=logging.DEBUG)


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def get_matching(args, df, imgWidth, imgHeight):

    for i in range(len(df)):
        preds2d = []
        imgPath = os.path.join(args.imgFolder, df.iloc[i]['imgPath'])

        for pNum, joints3d in enumerate(df.iloc[i]['pred']):
            try:
                pred2d = df.iloc[i, df.columns.get_loc('pred')][pNum]['joints']
            except BaseException:
                logging.fatal(
                    'pred joints  not available for ',
                    df.iloc[i]['pred_path'])

            preds2d.append(pred2d.squeeze())

        validDF = (np.array(df.iloc[i, df.columns.get_loc('occlusion')]) <= 90)\
            & (np.array(df.iloc[i, df.columns.get_loc('occlusion')]) >= 0)\
            & df.iloc[i, df.columns.get_loc('isValid')]

        matching = match_2d_greedy(imgWidth,
                                   imgHeight,
                                   pred_kps=preds2d,
                                   gtkp=df.iloc[i,
                                                df.columns.get_loc('gt_joints_2d')],
                                   debug_path = args.debug_path,
                                   imgPath=imgPath,
                                   baseline=args.baseline,
                                   valid=validDF,
                                   ind=i,
                                   debug=args.debug)

        # contains the matching between openpose and gt. each tuple is of order(idx_openpose_pred, idx_gt_kps)
        # this matching can be used to match pred to gt_kps, since the order of
        # openpose_pred and pred is the same
        df.iloc[i, df.columns.get_loc('matching')] = matching

    return df


# Todo work on this function and make it less confusing
def match_2d_greedy(
        imgWidth,
        imgHeight,
        pred_kps,
        gtkp,
        debug_path,
        imgPath=None,
        baseline=None,
        iou_thresh=0.1,
        valid=None,
        ind=-1,
        debug=False):
    '''
    matches groundtruth keypoints to the detection by considering all possible matchings.
    :return: best possible matching, a list of tuples, where each tuple corresponds to one match of pred_person.to gt_person.
            the order within one tuple is as follows (idx_pred_kps, idx_gt_kps)
    '''

    nkps = 24

    if debug and ind % 200 == 0:
        # To verify if projection of 2d joints on image are correct
        scatter2d(gtkp, pred_kps, imgPath, debug_path, baseline)

    predList = np.arange(len(pred_kps))
    gtList = np.arange(len(gtkp))
    # get all pairs of elements in pred_kps, gtkp
    # Todo optimize this approach
    # all combinations of 2 elements from l1 and l2
    combs = list(product(predList, gtList))

    errors_per_pair = {}
    errors_per_pair_list = []
    for comb in combs:
        errors_per_pair[str(comb)] = l2_error(
            pred_kps[comb[0]][:nkps, :2], gtkp[comb[1]][:nkps, :2])
        errors_per_pair_list.append(errors_per_pair[str(comb)])

    gtAssigned = np.zeros((len(gtkp),), dtype=bool)
    opAssigned = np.zeros((len(pred_kps),), dtype=bool)
    errors_per_pair_list = np.array(errors_per_pair_list)

    bestMatch = []
    excludedGtBecauseInvalid = []
    falsePositiveCounter = 0
    while np.sum(gtAssigned) < len(gtAssigned) and np.sum(
            opAssigned) + falsePositiveCounter < len(pred_kps):
        found = False
        falsePositive = False
        while not(found):
            if sum(np.inf == errors_per_pair_list) == len(
                    errors_per_pair_list):
                logging.fatal('something went wrong here')

            minIdx = np.argmin(errors_per_pair_list)
            minComb = combs[minIdx]
            # compute IOU
            iou = get_bbx_overlap(
                pred_kps[minComb[0]], gtkp[minComb[1]], imgPath, baseline)
            # if neither prediction nor ground truth has been matched before and iou
            # is larger than threshold
            if not(opAssigned[minComb[0]]) and not(
                    gtAssigned[minComb[1]]) and iou >= iou_thresh:
                logging.info(imgPath + ': found matching')
                found = True
                errors_per_pair_list[minIdx] = np.inf
            else:
                errors_per_pair_list[minIdx] = np.inf
                # if errors_per_pair_list[minIdx] >
                # matching_threshold*headBboxs[combs[minIdx][1]]:
                if iou < iou_thresh:
                    logging.info(
                        imgPath + ': false positive detected using threshold')
                    found = True
                    falsePositive = True
                    falsePositiveCounter += 1

        # if ground truth of combination is valid keep the match, else exclude
        # gt from matching
        if not(valid is None):
            if valid[minComb[1]]:
                if not falsePositive:
                    bestMatch.append(minComb)
                    opAssigned[minComb[0]] = True
                    gtAssigned[minComb[1]] = True
            else:
                gtAssigned[minComb[1]] = True
                excludedGtBecauseInvalid.append(minComb[1])

        elif not falsePositive:
            # same as above but without checking for valid
            bestMatch.append(minComb)
            opAssigned[minComb[0]] = True
            gtAssigned[minComb[1]] = True

    # add false positives and false negatives to the matching
    # find which elements have been successfully assigned
    opAssigned = []
    gtAssigned = []
    for pair in bestMatch:
        opAssigned.append(pair[0])
        gtAssigned.append(pair[1])
    opAssigned.sort()
    gtAssigned.sort()

    # handle false positives
    opIds = np.arange(len(pred_kps))
    # returns values of oIds that are not in opAssigned
    notAssignedIds = np.setdiff1d(opIds, opAssigned)
    for notAssignedId in notAssignedIds:
        bestMatch.append((notAssignedId, 'falsePositive'))
    gtIds = np.arange(len(gtList))
    # returns values of gtIds that are not in gtAssigned
    notAssignedIdsGt = np.setdiff1d(gtIds, gtAssigned)

    # handle false negatives/misses
    for notAssignedIdGt in notAssignedIdsGt:
        if not(valid is None):  # if using the new matching
            if valid[notAssignedIdGt]:
                logging.info(imgPath + ': miss')
                bestMatch.append(('miss', notAssignedIdGt))
            else:
                excludedGtBecauseInvalid.append(notAssignedIdGt)
        else:
            logging.info(imgPath + ': miss')
            bestMatch.append(('miss', notAssignedIdGt))

    # handle invalid ground truth
    for invalidGt in excludedGtBecauseInvalid:
        bestMatch.append(('invalid', invalidGt))

    return bestMatch  # tuples are (idx_pred_kps, idx_gt_kps)


def get_bbx_overlap(p1, p2, imgpath, baseline=None):
    if baseline.lower() == 'meanposebaseline':
        p1 = p1[p1[:, 2] > 0, :-1]

    min_p1 = np.min(p1, axis=0)
    min_p2 = np.min(p2, axis=0)
    max_p1 = np.max(p1, axis=0)
    max_p2 = np.max(p2, axis=0)

    bb1 = {}
    bb2 = {}

    bb1['x1'] = min_p1[0]
    bb1['x2'] = max_p1[0]
    bb1['y1'] = min_p1[1]
    bb1['y2'] = max_p1[1]
    bb2['x1'] = min_p2[0]
    bb2['x2'] = max_p2[0]
    bb2['y1'] = min_p2[1]
    bb2['y2'] = max_p2[1]

    try:
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']
    except BaseException:
        logging.fatal('why')

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = max(0, x_right - x_left + 1) * \
        max(0, y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1'] + 1) * (bb1['y2'] - bb1['y1'] + 1)
    bb2_area = (bb2['x2'] - bb2['x1'] + 1) * (bb2['y2'] - bb2['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou


def scatter2d(gtkp, openpose, imgPath, debug_path, baseline):
    import cv2
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    if not os.path.exists(debug_path):
        os.makedirs(debug_path)
    img = cv2.imread(imgPath)
    img = img[:, :, ::-1]

    colors = cm.tab20c(np.linspace(0, 1, 25))
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax.imshow(img)
    ax2.imshow(img)

    nkps = 24

    for pNum in range(len(openpose)):
        logging.debug(openpose[pNum].shape)
        for i in range(nkps):
            ax.scatter(openpose[pNum][i, 0], openpose[pNum]
                       [i, 1], c=colors[i], s=0.05)

    for pNum in range(len(gtkp)):
        for i in range(nkps):
            ax2.scatter(gtkp[pNum][i, 0], gtkp[pNum]
                    [i, 1], c=colors[i], s=0.05)

    if not (imgPath is None):
        savename = imgPath.split('/')[-1]
        savename = savename.replace('.pkl', '.jpg')
        plt.savefig(os.path.join(debug_path, baseline.lower() + savename))
        plt.close('all')
    logging.info('a')

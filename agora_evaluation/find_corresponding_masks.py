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
import pandas
import os

def getPersonMaskPath(maskBaseFolder, imgName, pnum, imgRes):
    scene_type='_'.join(imgName.split('_')[0:5])
    maskFolder = os.path.join(maskBaseFolder,scene_type)
    if imgRes=='high':
        imgBase = '_'.join(imgName.split('_')[0:-1])
        ending = '.png'
    elif imgRes=='low':
        imgBase = '_'.join(imgName.split('_')[0:-1])
        ending = '_1280x720.png'
    else:
        raise KeyError('imgRes can be either high or low')

    i = int(imgName.replace('.png','').split('_')[-1])
    format_pnum = format(pnum+1, '05d')
    format_i = format(i, '06d')
    if 'archviz' in imgName:
        cam = imgBase.split('_')[-1]
        imgBase = '_'.join(imgName.split('_')[0:-2])
        maskPerson = os.path.join(maskFolder, imgBase+'_mask_'+cam+'_'+format_i+'_'+format_pnum+ending)
    else:
        maskPerson = os.path.join(maskFolder, imgBase+'_mask_'+format_i+'_'+format_pnum+ending)
    print(maskPerson)
    return maskPerson

def find_corresponding_masks(*args):
    # Parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataframe_path', type=str,
                        default='')
    parser.add_argument('--maskBaseFolder', type=str,
                        default='')
    parser.add_argument(
        '--imgRes',
        type=str,
        default='high',
        help='high or low')
    args = parser.parse_args(*args)
    dataframe_path = args.dataframe_path
    maskBaseFolder = args.maskBaseFolder
    imgRes = args.imgRes
    df = pandas.read_pickle(dataframe_path)
    for i in range(len(df)):
         for pnum, gt in enumerate(df.iloc[i, df.columns.get_loc('isValid')]):
             maskPerson_path = getPersonMaskPath(maskBaseFolder, df.iloc[i]['imgPath'], pnum, imgRes)
             if os.path.exists(maskPerson_path):
                print('individual person mask: ',maskPerson_path)
             else:
                raise KeyError(maskPerson_path," Mask not found in maskBaseFolder")
         maskPerson_path = getPersonMaskPath(maskBaseFolder, df.iloc[i]['imgPath'], -1, imgRes)
         if os.path.exists(maskPerson_path):
             print('full mask: ', maskPerson_path)
         else:
             raise KeyError(maskPerson_path, " Mask not found in maskBaseFolder")

        
if __name__=='__main__':
    find_corresponding_masks(sys.argv[1:])

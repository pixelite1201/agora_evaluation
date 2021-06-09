# Evaluation Protocol:

Evaluating a method is done in 2 steps:

1. Matching Ground Truth with Predictions

2. Calculating error for corresponding matches

## Matching:
Since each image contains multiple person, to match the corresponding prediction with ground truth, we use projected joints in the image. We compute the joints error for all pair of matches and one with minimum error are considered a match. We use bounding boxes to make sure that the matched prediction and ground truth IOU < 0.1. If there is no match found for a particular ground truth, it is included in false negative. Please see the [paper](https://arxiv.org/abs/2104.14643) for more details.

We use the **joints** parameter in prediction dictionary .pkl file for predicted joints and **gt_joints_2d** parameter in dataframe .pkl file for  ground truth joints to perform matching. 

## Calculating Error:
Once we have the corresponding matches, we use different evaluation metric to perform the evaluation. 

MPJPE (Mean per joint position error): We evaluate joint error after aligning the root joint of prediction and ground truth. Root joint is pelvis joint for body, wrist joint for hands and neck joint for face.

MVE (Mean vertex error): We evaluate vertex error after aligning the root joint of prediction and ground truth. Root joint is pelvis joint for body, wrist joint for hands and neck joint for face. 

NMJE (Normalized mean joint error):  We normalize the MPJPE error by the standard detection metric, F1 score (the harmonic mean of recall and precision) to penalize methods for misses and false positives

NMVE (Normalized mean vertex error):  We normalize the MVE error by the standard detection metric, F1 score (the harmonic mean of recall and precision) to penalize methods for misses and false positives

### SMPL:

1. B-MPJPE: Evaluated on 24 body joints of SMPL after aligning the pelvis joint.

2. B-MVE: Evaluated on body vertices of SMPL after aligning the pelvis joint.

3. B-NMJE, B-NMVE: Normalize B-MPJPE and B-MVE by F1 score.

### SMPL-X:

1. B-MPJPE: Evaluated on 22 body joints of SMPL-X after aligning the pelvis joint.

2. B-MVE: Evaluated on body vertices of SMPL-X after aligning the pelvis joint. 

3. LH-MPJPE, RH-MPJPE: Evaluated on 15 hand joint after aligning the wrist joint.

4. LH-MVE, RH-MVE: Evaluated on MANO hand vertices after aligning the wrist joint.

4. F-MPJPE: Evaluated on 51 facial landmarks after aligning the neck joint.

5. F-MVE: Evaluated on FLAME head and face vertices after aligning the neck joint.

6. FB-MPJPE, FB-MVE: Weighted sum of the above B, LH, RH and F errors. Since the number of hand joints and face landmarks outweigh the number of body joints, we define the FB error as FB = B + (LH+RH+F)/

7. B-NMJE, FB-NMJE, B-NMVE, FB-NMVE: Normalized B-MPJPE, FB-MPJPE, B-MVE and FB-MVE by F1 score.



## Analysis:
We also provide detail analysis with respect to percentage of occlusion, distance of subject from the center of the image and subject orientation with respect to the camera. To see how well your method performs across these dimensions you could check the *_occlusion.png, *_xlocation.png and *_orientation.png plot.






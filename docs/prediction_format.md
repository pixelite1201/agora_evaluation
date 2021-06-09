# Prediction file format

Please read this page carefully to make sure that output prediction file format is compatible with the evaluation code. Sample prediction files are also provided in folder sample_pred_format. Please check corresponding [ReadMe](sample_pred_format/ReadMe.md) for details.

There can be two ways of submitting the prediction results:

1. Directly provide SMPL/SMPL-X 3D joints and vertices. This is easy but it will make the upload data size large and you might have to wait long to get the evaluation result.

2. Provide SMPL/SMPL-X parameters and the 3D joints and vertices will be generated on the server. The data size will be small and it will be quick to upload but you need to take care of the parameters shape and type.

We also provide the [code](agora_evaluation/check_pred_format.py) to verify that the prediction results are in correct format. Please see the instructions on how to run it on your zip file at the bottom of this page.

## Output filename format:
For each predicted person in an image, a dictionary should be generated and stored as pickle file in demo/predictions folder with following filename format.

If the image name is **Image.png** and there are 3 prediction for the corresponding image then the output prediction file name will be **Image_personId_0.pkl**, **Image_personId_1.pkl** and **Image_personId_2.pkl**


## SMPL-X dictionary (uploading joints and vertices):
For each predicted person in the image, a dictionary with following keys needs to be generated. **Note that the data type of all the parameters is np.ndarray.**

**joints** : (shape : (24,2), units : pixel coordinate). 2d projected joints location in the image. This is used to match the predition with the ground truth.

**verts** : (shape : (10475,3), units : meters). 3d vertices in camera coordinates. This is used to calculate the MVE/NMVE error for body, face and hands after aligning the root joint of prediction and ground truth.

**allSmplJoints3d** : (shape : (127, 3), units : meters). 3d joints in camera coordinates. This is used to calculate the MPJPE/NMJE error for body, face and hands after aligning the root joint of prediction and ground truth.

## SMPL-X dictionary (uploading parameters):
For each predicted person in the image, a dictionary with following keys needs to be generated. **Note that the data type of all the parameters is np.ndarray.**

**gender** : (Optional) male/female/neutral. If no gender is provided neutral will be used as default.

**age** : (Optional) adult/kid. If no age is provided adult will be used as default.

**num_betas** : (Optional) 10-300. If num_betas not provided, 10 betas will be used as default.

**pose2rot** : True/False. If pose2rot not provided, True will be used as default and pose parameters are expected in vector format. If the pose parameters are in rotation matrix format then pose2rot must be set to False.

**joints** : (shape : (24,2), units : pixel coordinate). 2d projected joints location in the image. This is used to match the predition with the ground truth.

**params** : Following parameters for SMPL-X models should be provided in dictionary form. Note that if **pose2rot** is True then pose parameters are in vector form and if it is False then pose parameters must be in matrix form. Corresponding shape for vector and matrix form are denoted by v_shape and m_shape.

> **transl** : (v_shape : (1,3), m_shape : (1,3))

> **betas** : (v_shape : (1,num_betas), m_shape : (1,num_betas))

> **expression** : (v_shape : (1,10), m_shape : (1,10))

> **global_orient** : (v_shape : (1,3), m_shape : (1,1,3,3))

> **body_pose** : (v_shape : (1,63), m_shape : (1,21,3,3))

> **left_hand_pose** : (v_shape : (1,45), m_shape : (1,15,3,3))

> **right_hand_pose** : (v_shape : (1,45), m_shape : (1,15,3,3))

> **leye_pose** : (v_shape : (1,3), m_shape : (1,1,3,3))

> **reye_pose** : (v_shape : (1,3), m_shape : (1,1,3,3))

> **jaw_pose** : (v_shape : (1,3), m_shape : (1,1,3,3))

### Very important: 
If you used 1280x720 resolution images **joints** field should always be converted to 3840x2160 format before submitting on evaluation server. To do this:

```
joints_4k = joints_720 *(2160/720)
```

## SMPL dictionary (uploading joints and vertices):
For each predicted person in the image a dictionary with following information needs to be generated. **Note that the data type of all the parameters is np.ndarray.**

**joints** : (shape : (24,2), units : pixel coordinate). 2d projected joints location in the image. This is used to match the predition with the ground truth.

**verts** : (shape : (6890,3), units : meters). 3d vertices in camera coordinates. This is used to calculate the MVE/NMVE error after aligning the root joint of prediction and ground truth.

**allSmplJoints3d** : (shape : (24, 3), units : meters). 3d joints in camera coordinates. This is used to calculate the MPJPE/NMJE error after aligning the root joint of prediction and ground truth.

## SMPL dictionary (uploading parameters):

For SMPL evaluation, neutral gender and 10 betas are used by default. **Note that the data type of all the parameters is np.ndarray.**

**age** : (Optional) adult/kid. If no age is provided adult will be used as default.

**pose2rot** : True/False. If pose2rot not provided, True will be used as default and pose parameters are expected in vector format. If the params are in rotation matrix format then pose2rot must be set to False.

**joints** : (shape : (24,2), units : pixel coordinate). 2d projected joints location in the image

**params** : Following parameters for SMPL-X models should be provided in dictionary form. Following parameters for SMPL-X models should be provided in dictionary form. Note that if **pose2rot** is True then pose parameters are in vector form and if it is False then pose parameters must be in matrix form. Corresponding shape for vector and matrix form are denoted by v_shape and m_shape.

> **transl** : (v_shape : (1,3), m_shape : (1,3))

> **betas** : (v_shape : (1,num_betas), m_shape : (1,num_betas))

> **global_orient** : (v_shape : (1,1,3), m_shape : (1,1,3,3))

> **body_pose** : (v_shape : (1,23,3), m_shape : (1,23,3,3))

### Very important: 
If you used 1280x720 resolution images for prediction, **joints** field should always be converted to 3840x2160 format before submitting. To do this:

```
joints_4k = joints_720 *(2160/720)
```

## Run check_pred_format
Once you have generated all the prediction (.pkl) files as explained above, create a zip of the folder containg the files e.g. pred.zip. The following command will extract the pred.zip in extract_zip folder and will verify if the shape and type for all the parameters in the individual pickle file is correct. 
```
python agora_evaluation/check_pred_format.py --predZip pred.zip --extractZipFolder extract_zip --modeltype SMPLX/SMPL
```

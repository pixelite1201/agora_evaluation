# Ground truth dataframe
Ground truth dataframe consists of all the information corresponding to the images in the dataset e.g. camera, joints, vertices, ground truth fit path etc.
Download the validation Camera dataframe (with SMPL joints and vertices/with SMPLX joints and vertices) from [AGORA Downloads](https://agora.is.tue.mpg.de/download.php) and store in demo/gt_dataframe folder. Details about different fields of the dataframe are as follows:

**imgPath** : Name of the image

**X, Y, Z** : Location of the scan in world coordinates

**Yaw**: Rotation of the scan along y axis

**camX, camY, camZ, camYaw** : Location of the camera in world coordinates

**gender** : gender of the scan

**kid** : adult or kid flag

**occlusion** : percentage of occlusion

**isValid** : If the occlusion is in range 0-90, isValid is True else False.

**gt_path_smpl, gt_path_smplx** : SMPL and SMPLX ground truth for each scan

**gt_joints_2d** : projected 2d SMPL or SMPLX keypoints/joints in image

**gt_joints_3d** : 3d SMPL or SMPLX joints in camera coordinates

**gt_verts** : 3d SMPL or SMPLX vertices in camera coordinates


## For 1280x720

Note that the above dataframe has **gt_joints_2d** field corresponding to 3840x2160 image resolution. The corresponding projection for 1280x720 is easy to generate using:
```
df_720[['gt_joints_2d'] = df_4k['gt_joints_2d']*(720/2160)
```

## Project Joints/Vertices
Note that the Camera dataframe (with SMPL joints and vertices/with SMPLX joints and vertices) already contains the joints/vertices. 
If you just want to project the joints and vertices using the SMPL/SMPL-X parameter file and Camera information then you need to run `project_joints` executable. Please check [Project Joints and Vertices](docs/project_joints_vertices.md).
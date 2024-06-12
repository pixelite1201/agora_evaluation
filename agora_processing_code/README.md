# Introduction
Following repository consists code to convert the ground truth SMPL-X files (neutral) provided on AGORA website in a format use for training Human pose and shape estimiation methods. One could modify the code to generate output in desire format for training.

# Prepare data
Run following script to download data. Note that you need to register on [SMPL-X](https://smpl-x.is.tue.mpg.de/) and [AGORA](https://agora.is.tue.mpg.de/) website to get access to the data
```
bash fetch_data.sh
```
Alternatively, you could manually dowload the files from the website.
Please note that all the downloaded images should be combined in single folder data/images 

# Run

```
python dataframe_agora_smplx_release.py

```


# Generated npz file format
```
imgname = Name of the images
center = center of the subject
scale = scale of the subject
pose_cam = SMPL-X pose parameter of the subject in camera coordinates
pose_world = SMPL-X pose parameter of the subject in world coordinates
shape = 11 SMPL-X shape componenets
gtkps = 2d joints
cam_int = camera intrinsic matrix
cam_ext = camera extrinsic matrix
proj_verts = 2d vertices subsampled
bfh = body or bfh 
```
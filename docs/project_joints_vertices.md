# Project Joints and Vertices using Camera Parameters

If you just want to project the joints and vertices using the SMPL/SMPL-X parameter file and Camera information then you need to run `project_joints`. Following page will guide you through it.

## Prerequisites
Create and activate a `Python 3.8` virtual environment:
```
python 3.8 -m venv path_to_virtual_env
source path_to_virtual_env/bin/activate
```

## Installation
First, checkout the code and its submodule [smplx](https://github.com/jcpassy/smplx):
```
$ git clone --recurse-submodules https://gitlab.tuebingen.mpg.de/ppatel/agora_evaluation.git
```

and install both packages with `pip`:
```
$ pip install .
$ pip install ./smplx
```
## Download

### Download the AGORA Images:
Download the [Images](https://agora.is.tue.mpg.de/) and extract them in demo/images

### Download the AGORA Camera dataframe:
Download the [Camera dataframe](https://agora.is.tue.mpg.de/) and extract in demo/Cam

### For SMPL-X projection:
Download the [SMPL-X fits](https://agora.is.tue.mpg.de/) and extract it in demo/GT_fits

Download the [SMPL-X](https://smpl-x.is.tue.mpg.de/) model and place it in demo/model/smplx. Download the npz version and rename the models to SMPLX_MALE.npz, SMPLX_FEMALE.npz and SMPLX_NEUTRAL.npz

Download the kid_template_file from where??

### For SMPL projection:
Download the [SMPL fits](https://agora.is.tue.mpg.de/) model and extract it in demo/GT_fits

Download the [SMPL](https://smpl.is.tue.mpg.de/) model and place it in demo/model/smpl. 
Download the npz version and rename the models to SMPL_MALE.npz, SMPL_FEMALE.npz, SMPL_NEUTRAL.npz

Download the kid_template_file from where??

# Run
Once you have finished installing and downloaded the data, you can run `project_joints` executable. This will add following fields in all the .pkl files in the demo/Cam folder.

**gt_joints_2d** : projected 2d SMPL -X/SMPL keypoints/joints in image

**gt_joints_3d** : 3d SMPL-X/SMPL joints in camera coordinates

**gt_verts** : 3d SMPL-X/SMPL vertices in camera coordinates

### SMPL-X

```
project_joints --imgFolder demo/images --loadPrecomputed demo/Cam --modeltype SMPLX --kid_template_path template/smplx_kid_template.obj  --modelFolder demo/model --gt_model_path demo/GT_fits/ --imgWidth 3840 --imgHeight 2160

```

### SMPL
```
project_joints --imgFolder demo/images --loadPrecomputed demo/Cam --modeltype SMPL --kid_template_path template/smpl_kid_template.obj  --modelFolder demo/model --gt_model_path demo/GT_fits/ --imgWidth 3840 --imgHeight 2160

```
#### For 1280x720:
Just replace the --imgWidth and --imgHeight with 1280 and 720 in above commands.
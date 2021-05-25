# Evaluation on AGORA

Following repository contains code to evaluate on AGORA validation/test images.
If you just want to project the joints and vertices using the SMPL/SMPL-X parameter file and Camera information then you need to run `project_joints` executable. Please check [Project Joints and Vertices](docs/project_joints_vertices.md).

# Prerequisites
Create and activate a `Python 3.8` virtual environment:
```
python 3.8 -m venv path_to_virtual_env
source path_to_virtual_env/bin/activate
```

# Installation
First, checkout the code and its submodule [smplx](https://github.com/jcpassy/smplx):
```
$ git clone --recurse-submodules https://gitlab.tuebingen.mpg.de/ppatel/agora_evaluation.git
```

and install both packages with `pip`:
```
$ pip install .
$ pip install ./smplx
```

# Downloads
## SMPL-X/SMPL model download
Download and rename the models to SMPL_MALE.npz, SMPL_FEMALE.npz, SMPL_NEUTRAL.npz (for smpl models) and SMPLX_MALE.npz, SMPLX_FEMALE.npz and SMPLX_NEUTRAL.npz (for smplx models)

### For SMPL-X evaluation:
Download the [SMPL-X](https://smpl-x.is.tue.mpg.de/) model and place it in demo/model/smplx

### For SMPL evaluation:
Download the [SMPL](https://smpl.is.tue.mpg.de/) model and place it in demo/model/smpl

## SMPL-X/SMPL vertex indices download
Does one need to download these from website? What about body vertices?They are not available

## Kid template model download
Should we provide it with repo or add it somewhere? 

## Ground truth dataframe
Check [Ground Truth dataframe](docs/gt_dataframe.md) to get more details about different dataframes and how to use them.

# Preparation of prediciton data for evaluation
Check [Predction format](docs/prediction_format.md) to get more details about the format for the output prediction file.

# Evaluate predictions
To run the evaluation for SMPL-X results, use the `evaluate_agora` executable:
```
$ evaluate_agora --pred_path demo/predictions/ --result_savePath demo/results/ --imgFolder demo/images/ --loadPrecomputed demo/gt_dataframe/  --baseline demo_model --modeltype SMPLX --indices_path utils --kid_template_path template/smplx_kid_template.obj --modelFolder demo/model/ --onlybfh

```
To run the evaluation for SMPL results, use the `evaluate_agora` executable:
```
$ evaluate_agora --pred_path demo/predictions_smpl/ --result_savePath demo/results/ --imgFolder demo/images/ --loadPrecomputed demo/gt_dataframe_smpl/  --baseline demo_model --modeltype SMPL --indices_path utils --kid_template_path template/smpl_kid_template.obj  --modelFolder demo/model/

```


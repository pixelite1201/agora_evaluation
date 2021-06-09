This page contains information about how to use the code to evaluate on AGORA validation/test images. 

If you are interested in other sections please see the following pages:

1. [Project joints and vertices](docs/project_joints_vertices.md):
Project the joints and vertices using the SMPL/SMPL-X parameter file and camera information.
2. [Find corresponding masks](docs/corresponding_masks.md): Find corresponding masks for images.
3. [Check prediction file format](docs/prediction_format.md): Check format of the prediction files before submitting for evaluation
4. [Evaluation metric and protocol](docs/evaluation_metric.md): Details about the evaluation metric and protocols.
5. [How to use kid model](docs/kid_model.md): Details on how to use kid model with SMPL-X.

# Evaluation on AGORA
If you want to evaluate the results of your 3D human pose and shape estimation method on AGORA validation images, you can follow the following steps. It is highly recommended to run the evaluation on validation images before submitting the results on test images on evaluatoin server. 


## Prerequisites
Create and activate a `Python 3.8` virtual environment:
```
python3.8 -m venv path_to_virtual_env
source path_to_virtual_env/bin/activate
```

## Installation
First, checkout the code [smplx](https://github.com/vchoutas/smplx.git):
```
$ git clone https://github.com/vchoutas/smplx.git
```

and install both packages with `pip`:
```
$ pip install .
$ pip install ./smplx
```

## Downloads
### SMPL-X/SMPL model download

#### For SMPL-X evaluation:
Download the [SMPL-X](https://smpl-x.is.tue.mpg.de/) model and place the model files in demo/model/smplx. 
Rename the model files SMPLX_MALE.npz, SMPLX_FEMALE.npz and SMPLX_NEUTRAL.npz if needed.

#### For SMPL evaluation:
Download the [SMPL](https://smpl.is.tue.mpg.de/) model and place it in demo/model/smpl
Rename the models to SMPL_MALE.pkl, SMPL_FEMALE.pkl, SMPL_NEUTRAL.pkl if needed.

### SMPL-X/SMPL vertex indices download
Download the vertex indices from [SMPL-X](https://smpl-x.is.tue.mpg.de/) from the section **MANO and FLAME vertex indices** and place it in utils.

### SMPL-X/SMPL kid template model download
Download the kid template vertices from [AGORA](https://agora.is.tue.mpg.de) and place them in utils.

### Ground truth dataframe download
Ground truth dataframe consists of all the information corresponding to the images in the dataset e.g. camera, joints, vertices, ground truth fit path etc.
Download the validation Camera dataframe (with SMPL joints and vertices or with SMPLX joints and vertices) from [AGORA Downloads](https://agora.is.tue.mpg.de/download.php) and extract all the .pkl files in demo/gt_dataframe folder.
Check [Ground Truth dataframe](docs/gt_dataframe.md) to get more details about different dataframes and how to use them.

## Preparation of prediciton data for evaluation
In short for each predicted person in an image, a dictionary should be generated and stored as pickle file in demo/predictions folder.
Please check [Predction format](docs/prediction_format.md) to get more details about the format for the output prediction file. Please go through the page carefully. If the output format for prediction file is not correct, the evaluation pipeline will fail.

## Evaluate predictions

To run the evaluation for SMPL-X results, use the `evaluate_agora` executable:
```
$ evaluate_agora --pred_path demo/predictions/ --result_savePath demo/results/ --imgFolder demo/images/ --loadPrecomputed demo/gt_dataframe/  --baseline demo_model --modeltype SMPLX --indices_path utils --kid_template_path utils/smplx_kid_template.npy --modelFolder demo/model/ --onlybfh

```
To run the evaluation for SMPL results, use the `evaluate_agora` executable:
```
$ evaluate_agora --pred_path demo/predictions_smpl/ --result_savePath demo/results/ --imgFolder demo/images/ --loadPrecomputed demo/gt_dataframe_smpl/  --baseline demo_model --modeltype SMPL --indices_path utils --kid_template_path utils/smpl_kid_template.npy  --modelFolder demo/model/

```

To run the evaluation for 1280x720 version, just provide --imgWidth 1280 and --imgHeight 720 as parameter to `evaluate_agora` executable. Note that the --imgFolder and --loadPrecomputed should also be replaced with 1280x720 version.

If you need to debug the projection of ground truth and prediction keypoints in image please provide '--debug' boolean flag and '--debug_path' as the path to output the images. This will generate images in debug_path where left part show the overlaid prediction keypoints and right part show the overlaid ground truth keypoints.

```
$ evaluate_agora --pred_path demo/predictions/ --result_savePath demo/results/ --imgFolder demo/images/ --loadPrecomputed demo/gt_dataframe/  --baseline demo_model --modeltype SMPLX --indices_path utils --kid_template_path utils/smplx_kid_template.npy --modelFolder demo/model/ --onlybfh --debug --debug_path demo/debug
```

# Citation
If you use this code, please cite:

```bibtex
@inproceedings{Patel:CVPR:2021,
  title = {{AGORA}: Avatars in Geography Optimized for Regression Analysis}, 
  author = {Patel, Priyanka and Huang, Chun-Hao P. and Tesch, Joachim and Hoffmann, David T. and Tripathi, Shashank and Black, Michael J.}, 
  booktitle = {Proceedings IEEE/CVF Conf.~on Computer Vision and Pattern Recognition ({CVPR})}, 
  month = jun,
  year = {2021},
  month_numeric = {6}
}
```

# References

Here are some great resources that we used:

[SMPL-X](https://smpl-x.is.tue.mpg.de/)

[SMPL](https://smpl.is.tue.mpg.de/)

[SMIL](https://www.iosb.fraunhofer.de/en/competences/image-exploitation/object-recognition/sensor-networks/motion-analysis.html)


# Acknowledgement
Special thanks to [Vassilis Choutas](https://ps.is.tuebingen.mpg.de/person/vchoutas) for sharing the pytorch code used in fitting the SMPL-X model to the scans.

# Contact
For questions, please contact agora@tue.mpg.de.

For commercial licensing (and all related questions for business applications), please contact ps-licensing@tue.mpg.de.

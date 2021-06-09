# Kid model

To use kid model with [SMPL-X](https://github.com/vchoutas/smplx.git) code, you need to download the kid template vertices from [AGORA](https://agora.is.tue.mpg.de/) Download page.

Then you can use the smplx api just by adding the flag age=='kid' and providing the path where you stored the downloaded kid template vertices in flag kid_template_path.
```
import smplx
model_smplx_kid = smplx.create(modelFolder, model_type='smplx', age='kid', kid_template_path=smplx_kid_template_path)
model_smpl_kid = smplx.create(modelFolder, model_type='smpl', age='kid', kid_template_path=smpl_kid_template_path)

```
If num_betas is set to **n** it will create a model with **n+1** betas where the **n+1th** shape component is the interpolation of SMPL/SMPL-X and SMIL/SMIL-X template.
Note that it is recommended to use gender= 'male' for male kids and gender='neutral' for female kids. 
# Corresponding Masks:
Given a dataframe with image names, if you want to find the corresponding masks for each image you can use the following code.

**--dataframe_path** - path to the dataframe **.pkl** file 

**--maskBaseFolder** - path to the base folder containing all the masks

**-**-imgRes** - high for 3840x2160 masks and low for 1280x720 masks

```
python find_corresponding_masks.py --dataframe_path  path_to_dataframe --maskBaseFolder path_to_masks --imgRes high/low
```

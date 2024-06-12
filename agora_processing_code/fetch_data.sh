#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# SMPL-X model
echo -e "\nYou need to register at https://smpl-x.is.tue.mpg.de"
read -p "Username (SMPL-X):" username
read -p "Password (SMPL-X):" password
username=$(urle $username)
password=$(urle $password)

mkdir -p data
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip' -O './data/smplx.zip' --no-check-certificate --continue
unzip data/smplx.zip -d data


# # # # #  Ground truth labels
echo -e "\nYou need to register at https://agora.is.tue.mpg.de/"
read -p "Username (AGORA):" username
read -p "Password (AGORA):" password
username=$(urle $username)
password=$(urle $password)

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=agora&resume=1&sfile=smplx_kid_template.npy' -O './data/smplx_kid_template.npy' --no-check-certificate --continue

wget   --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=agora&sfile=smplx_gt_neutral_gender.zip' -O './data/smplx_gt_neutral_gender.zip' --no-check-certificate --continue
unzip data/smplx_gt_neutral_gender.zip -d data

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=agora&resume=1&sfile=train_SMPLX.zip' -O './data/train_SMPLX.zip' --no-check-certificate --continue
unzip data/train_SMPLX.zip -d data/train_df

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=agora&resume=1&sfile=train_images_1280x720_0.zip' -O './data/train_images_1280x720_0.zip' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=agora&resume=1&sfile=train_images_1280x720_1.zip' -O './data/train_images_1280x720_1.zip' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=agora&resume=1&sfile=train_images_1280x720_2.zip' -O './data/train_images_1280x720_2.zip' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=agora&resume=1&sfile=train_images_1280x720_3.zip' -O './data/train_images_1280x720_3.zip' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=agora&resume=1&sfile=train_images_1280x720_4.zip' -O './data/train_images_1280x720_4.zip' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=agora&resume=1&sfile=train_images_1280x720_5.zip' -O './data/train_images_1280x720_5.zip' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=agora&resume=1&sfile=train_images_1280x720_6.zip' -O './data/train_images_1280x720_6.zip' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=agora&resume=1&sfile=train_images_1280x720_7.zip' -O './data/train_images_1280x720_7.zip' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=agora&resume=1&sfile=train_images_1280x720_8.zip' -O './data/train_images_1280x720_8.zip' --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=agora&resume=1&sfile=train_images_1280x720_9.zip' -O './data/train_images_1280x720_9.zip' --no-check-certificate --continue

echo -e "\n You need to combine all the images in single folder data/images"


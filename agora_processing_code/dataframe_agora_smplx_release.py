import os
import pickle
import torch
import cv2
import smplx
import pickle
import pandas
import argparse 

from renderer_pyrd import Renderer
import matplotlib.pyplot as plt
from glob import glob
import numpy as np


SCALE_FACTOR_BBOX = 1.2
IMG_WIDTH = 1280
IMG_HEIGHT = 720

downsample_mat = pickle.load(open('downsample_mat_smplx.pkl','rb'))

def focalLength_mm2px(focalLength, dslr_sens, focalPoint):
    focal_pixel = (focalLength / dslr_sens) * focalPoint * 2
    return focal_pixel

def unreal2cv2(points):
    # x --> y, y --> z, z --> x
    points = np.roll(points, 2, 1)
    # change direction of y
    points = points * np.array([1.0, -1.0, 1.0])
    return points


def get_cam_pitch_yaw(imgPath, camYaw):
    if 'hdri' in imgPath:
        camYaw=0
        camPitch=0
    elif 'cam00' in imgPath:
        camYaw=135
        camPitch=30
    elif 'cam01' in imgPath:
        camYaw=-135
        camPitch=30
    elif 'cam02' in imgPath:
        camYaw=-45
        camPitch=30
    elif 'cam03' in imgPath:
        camYaw=45
        camPitch=30
    else:
        camYaw = camYaw
        camPitch=0
    
    return camYaw, camPitch

def get_smplx_vertices(poses, betas, trans, age, model_dict):
    model = model_dict[age]
    model_out = model(betas=torch.tensor(betas).unsqueeze(0).float(),
                            global_orient=torch.tensor(poses[:3]).unsqueeze(0).float(),
                            body_pose=torch.tensor(poses[3:66]).unsqueeze(0).float(),
                            left_hand_pose=torch.zeros(poses[75:120].shape).unsqueeze(0).float(),
                            right_hand_pose=torch.zeros(poses[120:165].shape).unsqueeze(0).float(),
                            jaw_pose=torch.tensor(poses[66:69]).unsqueeze(0).float(),
                            leye_pose=torch.tensor(poses[69:72]).unsqueeze(0).float(),
                            reye_pose=torch.tensor(poses[72:75]).unsqueeze(0).float(),
                            transl=(trans).unsqueeze(0))

    return model_out.vertices[0], model_out.joints[0]



def visualize(image_path, verts,focal_length, smpl_faces, ind=0):

    img = cv2.imread(image_path)
    h,w,c = img.shape
    renderer = Renderer(focal_length=focal_length, img_w=w, img_h=h,
                            faces=smpl_faces)
    front_view = renderer.render_front_view(verts.unsqueeze(0).detach().cpu().numpy(),
                                    bg_img_rgb=img[:, :, ::-1].copy())
    cv2.imwrite(image_path.split('/')[-1].replace('.png', str(ind)+'.png'), front_view[:, :, ::-1])

def get_bbox_valid(joints, img_height, img_width, rescale):
    #Get bbox using keypoints
    #Convert joints to new img_width and height
    joints = np.copy(joints)

    valid_j = []
    for j in joints[:25]:
        if j[0] > img_width or j[1] > img_height or j[0] < 0 or j[1] < 0:
            continue
        else:
            valid_j.append(j)

    if len(valid_j) < 1:
        return [-1, -1], -1, len(valid_j), [-1, -1, -1, -1]

    joints = np.array(valid_j)

    bbox = [min(joints[:, 0]), min(joints[:, 1]), max(joints[:, 0]), max(joints[:, 1])]

    center = [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2]
    scale = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200

    scale *= rescale

    return center, scale, len(valid_j), bbox


def get_transform(center, scale, res, rot=0):
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0):
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int) + 1

def crop(img, center, scale, res, rot=0):
    # Upper left point

    ul = np.array(transform([1, 1], center, scale, res, invert=1)) - 1
    # Bottom right point
    br = np.array(transform([res[0] + 1,
                             res[1] + 1], center, scale, res, invert=1)) - 1

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]

    if len(img.shape) > 2:
        new_shape += [img.shape[2]]
    new_img = np.zeros(new_shape)

    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1],
                                                    old_x[0]:old_x[1]]

    if not rot == 0:
        new_img = rotate(new_img, rot) # scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]
    from skimage.transform import rotate, resize
    new_img = resize(new_img, res) # scipy.misc.imresize(new_img, res)
    return img,new_img


def visualize_2d(image_path, joints2d):
    from matplotlib import pyplot as plt

    img = cv2.imread(image_path)
    img = img[:, :, ::-1]
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    ax.imshow(img)
    for i in range(len(joints2d)):
        ax.scatter(joints2d[i, 0], joints2d[i, 1], s=0.05)
    plt.savefig(image_path.split('/')[-1])

def visualize_crop(image_path, center, scale, verts,focal_length, smpl_faces, ind):
    img = cv2.imread(image_path)
    h,w,c = img.shape

    renderer = Renderer(focal_length=focal_length, img_w=w, img_h=h,
                            faces=smpl_faces)
    front_view = renderer.render_front_view(verts.unsqueeze(0).detach().cpu().numpy(),
                                    bg_img_rgb=img[:, :, ::-1].copy())
    img, crop_img = crop(front_view[:, :, ::-1], center, scale, res=(224,224))
    cv2.imwrite(image_path.split('/')[-1].replace('.png',str(ind)+'.png'), crop_img)
    cv2.imwrite(image_path.split('/')[-1].replace('.png',str(ind)+'_full.png'), img)

def get_cam_rotmat( pitch, yaw, roll):
    #Because bodies are rotation by 90

    rotmat_yaw, _ = cv2.Rodrigues(np.array([[0, (( yaw) / 180) * np.pi, 0]], dtype=float))
    rotmat_pitch, _ = cv2.Rodrigues(np.array([pitch / 180 * np.pi, 0, 0]).reshape(3, 1))
    rotmat_roll, _ = cv2.Rodrigues(np.array([0, 0,roll / 180 * np.pi]).reshape(3, 1))
    final_rotmat = np.matmul(rotmat_roll, np.matmul(rotmat_pitch, rotmat_yaw))
    return final_rotmat


def project(points, cam_trans, cam_int):

    points = points + cam_trans
    cam_int = torch.tensor(cam_int).float()

    projected_points = points / points[:,-1].unsqueeze(-1)
    projected_points = torch.einsum('ij, kj->ki', cam_int, projected_points.float())

    return projected_points.detach().cpu().numpy()


def toWorldCamCoord(args, df, i, pNum, imgWidth, imgHeight, model_dict, visualize=False):
    gt_path_neutral = os.path.join(args.smplx_neutral_gt,  df.iloc[i]['gt_path_smplx'][pNum].replace('.obj','.pkl'))
    params = pandas.read_pickle(gt_path_neutral)
    
    global_orient = params['global_orient']
    body_pose = params['body_pose']
    left_hand_pose = params['left_hand_pose']
    right_hand_pose = params['right_hand_pose']
    jaw_pose = params['jaw_pose']
    leye_pose = params['leye_pose']
    reye_pose = params['reye_pose']
    expression = params['expression']
    betas = params['betas']
    translation = params['transl']

    dslr_sens_width = 36
    dslr_sens_height = 20.25
    cx = imgWidth/2
    cy = imgHeight/2
    imgPath = df.iloc[i]['imgPath']
    if 'hdri' in imgPath:
        ground_plane = [0, 0, 0]
        focalLength = 50
        camPosWorld = [0, 0, 170]

    elif 'cam00' in imgPath:
        ground_plane = [0, 0, 0]
        focalLength = 18
        camPosWorld = [400, -275, 265]

    elif 'cam01' in imgPath:
        ground_plane = [0, 0, 0]
        focalLength = 18
        camPosWorld = [400, 225, 265]

    elif 'cam02' in imgPath:
        ground_plane = [0, 0, 0]
        focalLength = 18
        camPosWorld = [-490, 170, 265]

    elif 'cam03' in imgPath:
        ground_plane = [0, 0, 0]
        focalLength = 18
        camPosWorld = [-490, -275, 265]
    else:
        ground_plane = [0, -1.7, 0]
        focalLength = 28
        camPosWorld = [
            df.iloc[i]['camX'],
            df.iloc[i]['camY'],
            df.iloc[i]['camZ']]

    yawSMPL = df.iloc[i]['Yaw'][pNum]
    trans3d = [df.iloc[i]['X'][pNum],
                df.iloc[i]['Y'][pNum],
                df.iloc[i]['Z'][pNum]]

    focalLength_x = focalLength_mm2px(focalLength, dslr_sens_width, cx)
    focalLength_y = focalLength_mm2px(focalLength, dslr_sens_height, cy)

    intrinsics = (focalLength_x, focalLength_y, cx, cy) 

    # camPosWorld and trans3d are in cm. Transform to meter
    trans3d = np.array(trans3d) / 100
    trans3d = unreal2cv2(np.reshape(trans3d, (1, 3)))
    camPosWorld = np.array(camPosWorld) / 100
    camPosWorld = unreal2cv2(np.reshape(camPosWorld, (1, 3))) + np.array(ground_plane)

    #get cam yaw and pitch 
    img_name = df.iloc[i]['imgPath']
    camYaw, cam_pitch = get_cam_pitch_yaw(img_name, df.iloc[i]['camYaw'])
    
    #World coordinate transformation after assuming camera has 0 yaw and is at origin
    rotMat, _ = cv2.Rodrigues(np.array([[0, ((yawSMPL - 90 - camYaw) / 180) * np.pi, 0]], dtype=float))
    yaw_rotMat, _ =cv2.Rodrigues(np.array([0, (-camYaw) / 180 * np.pi, 0]).reshape(3, 1))
    pitch_rotMat, _ = cv2.Rodrigues(np.array([cam_pitch / 180 * np.pi, 0, 0]).reshape(3, 1))
    transform_coordinate = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    transform_rotMat = np.matmul(rotMat, transform_coordinate)
    w_global_orient = cv2.Rodrigues(np.dot(transform_rotMat, cv2.Rodrigues(global_orient)[0]))[0].T[0]

    #Applying rotation to translation vector and pelvis joint (j0) as it is not at origin. 
    # This is how rotation is applied in SMPL
    # v_out = R_o(v_template - j0) + j0 --> pelvis is moved to origin, then rotated and then moved back
    # Case 1: Applying rotation to vetices:
    # v1 = R_x[R_o(v_template-j0) + j0] + transl
    # v1 = R_x.R_o.v_tempalte - R_x.R_o.j0 + R_x.j0 + transl
    # Case 2: Applying rotation to global orientation:
    # v2 = R_x.R_o(v_template-j0) + j0 + transl
    # v2 = R_x.R_o.v_tempalte - R_x.R_o.j0 + j0 + transl
    # --> Now to make v2 same as v1, we need to add following
    # diff = R_x.j0 - j0
    # --> so v2 should be and the diff is provided in transl
    # v2 =  R_x.R_o.v_tempalte - R_x.R_o.j0 + diff + transl

    #Since in AGORA adult shape param consist of 10 betas and kid shape param consist of 11 betas
    if df.iloc[i]['kid'][pNum]:
        model = model_dict['kid']
    else:
        model = model_dict['adult']

    gt_local = model(betas=torch.tensor(betas, dtype=torch.float),
            global_orient=torch.tensor(global_orient, dtype=torch.float),
            body_pose=torch.tensor(body_pose, dtype=torch.float),
            translation=torch.zeros((1,3), dtype=torch.float))

    j0 = gt_local.joints[0][0].detach().cpu().numpy()
    rot_j0 = np.matmul(transform_rotMat, j0.T).T

    l_translation_ = np.matmul(transform_rotMat, translation.T).T
    l_translation = rot_j0 + l_translation_
    cam_translation = trans3d - camPosWorld
    w_translation = l_translation + cam_translation - j0

    c_global_orient = cv2.Rodrigues(np.dot(pitch_rotMat, cv2.Rodrigues(w_global_orient)[0]))[0].T[0]
    yaw_pitch_rotMat = np.matmul(pitch_rotMat, yaw_rotMat)
    c_translation = np.matmul(pitch_rotMat, l_translation.T).T + np.matmul(yaw_pitch_rotMat, cam_translation.T).T - j0 

    body_pose = np.hstack([body_pose, jaw_pose, leye_pose, reye_pose, left_hand_pose, right_hand_pose])
    w_params = {'body_pose':body_pose,'global_orient':w_global_orient,'transl':w_translation,'betas':betas }
    c_params = {'body_pose':body_pose,'global_orient':c_global_orient,'transl':c_translation,'betas':betas }

    return w_params, c_params, cam_pitch, intrinsics

def save_npz(args, model_dict):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    out_file = os.path.join(args.output_dir, f'agora.npz')
    all_annot = sorted(glob(os.path.join(args.dataframe,'*.pkl')))
    imgnames_, betas_, poses_cam_, cam_ext_, cam_int_, gtkps, centers_, scales_, proj_verts, pose_world_, bfh = [], [], [], [], [], [], [], [], [], [], []
    for pkl_i,pkl_path in enumerate(all_annot):
        print('Processing ', pkl_path)
        df = pickle.load(open(pkl_path,'rb'))
        
        for i in range(len(df)):
            for j, is_valid in enumerate(df.iloc[i]['isValid']):
                if df.iloc[i]['occlusion'][j]>70:
                    continue

                if 'bfh' in df.iloc[i]['gt_path_smplx'][j]:
                    bfh_val=True
                else:
                    bfh_val=False

                kid_flag = df.iloc[i]['kid'][j]
                img_name = df.iloc[i]['imgPath']

                w_params, c_params, cam_pitch, intrinsics = toWorldCamCoord(args, df, i, j, IMG_WIDTH, IMG_HEIGHT, model_dict)
            
                all_kp2d = df.iloc[i]['gt_joints_2d'][j]
                CAM_INT = np.identity(3)
                cx = IMG_WIDTH/2
                cy = IMG_HEIGHT/2
                CAM_INT[0,0] = intrinsics[0]
                CAM_INT[1,1] = intrinsics[1]
                CAM_INT[0,2] = intrinsics[2]
                CAM_INT[1,2] = intrinsics[3]

                cam_t = c_params['transl']
                w_trans = w_params['transl']
                pitch = cam_pitch
                cam_rotmat = get_cam_rotmat(pitch, 0, 0)
                cam_ext = np.zeros((4,4))
                cam_ext[:3, :3] = cam_rotmat
                cam_ext_trans = np.concatenate([cam_t, np.array([[1]])],axis=1)          
                cam_ext[:, 3] = cam_ext_trans

                pose_cam =  np.hstack((c_params['global_orient'],c_params['body_pose'][0]))
                pose_world = np.hstack((w_params['global_orient'],w_params['body_pose'][0]))

                betas = w_params['betas'][0]

                if kid_flag:
                    age='kid'
                else:
                    age='adult'

                vertices3d, joints3d_ = get_smplx_vertices(pose_cam, betas, torch.zeros(3), age, model_dict)

                joints2d =  project(joints3d_.detach().cpu(), torch.tensor(cam_t), CAM_INT)

                vertices3d_downsample = downsample_mat.matmul(vertices3d).clone().detach().float()
                proj_verts_ = project(vertices3d_downsample, (cam_t), CAM_INT)


                center, scale, num_vis_joints, bbox = get_bbox_valid(joints2d[:25], rescale=SCALE_FACTOR_BBOX, img_width=IMG_WIDTH, img_height=IMG_HEIGHT)

                if center[0]<=0 or center[1]<=0 or scale<=0:
                    continue
                if num_vis_joints<8:
                    continue
                image_path = os.path.join(args.img_dir, img_name.replace('.png','_1280x720.png'))

                # verts_cam2 =  vertices3d.detach().cpu().numpy() + cam_t
                # visualize(image_path, torch.tensor(verts_cam2) ,  CAM_INT[0][0], model_dict['adult'].faces, i)
                #visualize_2d(image_path, proj_verts)

                imgnames_.append(img_name.replace('.png','_1280x720.png'))
                betas_.append(betas)
                poses_cam_.append(pose_cam)
                cam_ext_.append(cam_ext)
                cam_int_.append(CAM_INT)
                gtkps.append(joints2d)
                centers_.append(center)
                scales_.append(scale)
                pose_world_.append(pose_world)
                proj_verts.append(proj_verts_)
                bfh.append(bfh_val)

                    
    print(f'Saving {out_file}...')
    np.savez(
        out_file,
        imgname=imgnames_,
        center=centers_,
        scale=scales_,
        pose_cam=poses_cam_,
        pose_world=pose_world_,
        shape=betas_,
        gtkps=gtkps,
        cam_ext=cam_ext_,
        cam_int=cam_int_,
        proj_verts=proj_verts,
        bfh=bfh
        )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataframe', type=str, help='Path to all agora dataframe pkl files', default='data/train_df/SMPLX')
    parser.add_argument('--img_dir', type=str, help='Path to image directory 1280x720 consiting of all images in single folder', default='data/images')
    parser.add_argument('--smplx_neutral_gt', type=str, help='Path to SMPL-X files in neutral format', default='data/smplx_gt_neutral_gender')
    parser.add_argument('--model_path', type=str, help='Path to SMPL-X model files', default='data/models/smplx')
    parser.add_argument('--kid_template_path', type=str, help='Path to kid template', default='data/smplx_kid_template.npy')
    parser.add_argument('--output_dir', type=str, help='Path where to store output processed npz',default='output')

    args = parser.parse_args()

    smplx_model_neutral = smplx.create(args.model_path, model_type='smplx',
                                    gender='neutral',
                                    ext='npz',
                                    num_betas=10,
                                    use_pca=False)  

    smplx_model_kid = smplx.create(args.model_path, model_type='smplx',
                                    age='kid',
                                    kid_template_path=args.kid_template_path,
                                    gender='neutral',
                                    ext='npz',
                                    use_pca=False) 
    model_dict = {}
    model_dict['adult'] = smplx_model_neutral 
    model_dict['kid'] = smplx_model_kid 

 
    save_npz(args, model_dict)
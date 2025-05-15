import os,sys
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "EVA"))
sys.path.insert(0, os.path.join(ROOT, "utils"))
# path = "/mnt/fast_nas/xiyingjie/code/priorMD"
# sys.path.append(path)
# from model.comMDM import ComMDM
# from model.ori_mdm import ini_MDM

import numpy as np
import shutil
import torch
import utils.rotation_conversions as geometry
from utils.humanml3d import Convert_Pose_to_Joints3D
from utils.Convert_TRC_MOT import make_animation_matplot
import math

def distance_3d(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


humanact12_raw_offsets_new = np.array([[0,0,0],
                               [0.13501,0,0],
                               [-0.1301,0,0],
                               [0,0.12152,0],
                               [0,-0.44043,0],
                               [0,-0.43475,0],
                               [0,0.1801,0],
                               [0,-0.44902,0],
                               [0,-0.45521,0],
                               [0,0.04992,0],
                               [0,0,0.1535],
                               [0,0,0.1541],
                               [0,0.2646,0],
                               [0.173,0,0],
                               [-0.17239,0,0],
                               [0,0,0.08723],
                               [0,-0.09333,0],
                               [0,-0.09776,0],
                               [0,-0.29157,0],
                               [0,-0.288,0],
                               [0,-0.27232,0],
                               [0,-0.28626,0]])

def CalculateStandardSkeleton(joints):
    print("0-1:", distance_3d(joints[0,0,:],joints[0,1,:]))
    print("0-2:", distance_3d(joints[0,0,:],joints[0,2,:]))
    print("1-4:", distance_3d(joints[0,1,:],joints[0,4,:]))
    print("2-5:", distance_3d(joints[0,2,:],joints[0,5,:]))
    print("4-7:", distance_3d(joints[0,4,:],joints[0,7,:]))
    print("5-8:", distance_3d(joints[0,5,:],joints[0,8,:]))
    print("7-10:", distance_3d(joints[0,7,:],joints[0,10,:]))
    print("8-11:", distance_3d(joints[0,8,:],joints[0,11,:]))
    print("0-3:", distance_3d(joints[0,0,:],joints[0,3,:]))
    print("3-6:", distance_3d(joints[0,3,:],joints[0,6,:]))
    print("6-9:", distance_3d(joints[0,6,:],joints[0,9,:]))
    print("9-12:", distance_3d(joints[0,9,:],joints[0,12,:]))
    print("9-13:", distance_3d(joints[0,9,:],joints[0,13,:]))
    print("9-14:", distance_3d(joints[0,9,:],joints[0,14,:]))
    print("12-15:", distance_3d(joints[0,12,:],joints[0,15,:]))
    print("13-16:", distance_3d(joints[0,13,:],joints[0,16,:]))
    print("14-17:", distance_3d(joints[0,14,:],joints[0,17,:]))
    print("16-18:", distance_3d(joints[0,16,:],joints[0,18,:]))
    print("17-19:", distance_3d(joints[0,17,:],joints[0,19,:]))
    print("18-20:", distance_3d(joints[0,18,:],joints[0,20,:]))
    print("19-21:", distance_3d(joints[0,19,:],joints[0,21,:]))


def draw_3dpw_npy(npy_path, save_path):
    data = np.load(npy_path,allow_pickle=True).item()
    mean  = np.load(os.path.join("/mnt/fast_nas/xiyingjie/code/priorMDM/dataset/3dpw/", 'skel_Mean.npy'))
    std = np.load(os.path.join("/mnt/fast_nas/xiyingjie/code/priorMDM/dataset/3dpw/", 'skel_Std.npy'))

    for i in range(len(data)):
        data_sample = data[i]
        k=0
        sample, sample1 = data_sample['motion_0'][k].unsqueeze(0), data_sample['motion_1'][k].unsqueeze(0)
        canon0, canon1 = data_sample['canon_0'][k].unsqueeze(0),data_sample['canon_1'][k].unsqueeze(0)
        canon0, canon1 = canon0.squeeze().cpu(), canon1.squeeze().cpu()

        rot_from_x_to_0, rot_from_x_to_1 = geometry.rotation_6d_to_matrix(canon0[:6]).numpy(), geometry.rotation_6d_to_matrix(canon1[:6]).numpy()
        dis_from_ori_to_0, dis_from_ori_to_1 = canon0[6:9].numpy(), canon1[6:9].numpy()
        
        print("0000::",sample.shape)
        # 1.3 motion数据恢复初始格式，反向归一化
        person_1, person_2 = sample.squeeze().cpu().permute(1,0).reshape(-1,23,6), sample1.squeeze().cpu().permute(1,0).reshape(-1,23,6)
        # person_1, person_2 = sample, sample1
        # mean_, std_ = torch.from_numpy(mean).cpu(), torch.from_numpy(std).cpu()
        # epsilon = 1e-6
        # # 找到所有为零的元素并替换为epsilon
        # std_[std_ == 0] = epsilon
        # person_1, person_2 = person_1 * std_ + mean_,  person_2 * std_ + mean_ # [120,23,6]

        sample, ret = person_1[:,:22,:],  person_1[:,22,:3]
        sample1, ret1 = person_2[:,:22,:],  person_2[:,22,:3]

        # sample1 = sample.squeeze().cpu().permute(1,0).reshape(120,-1,6), sample1.squeeze().cpu().permute(1,0).reshape(120,-1,6)

        # 1.4 把motions转回成3Djoints坐标
        # ret, ret1 = sample[:,22,:3], sample1[:,22,:3]

        
        # ret, ret1 = ret*4, ret1*4
        motion, motion1 = sample[:,:22,:], sample1[:,:22,:]
        axis_angle, axis_angle1 = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(motion)),  geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(motion1))

        joints, joints1 = Convert_Pose_to_Joints3D(axis_angle.numpy(), ret.numpy()), Convert_Pose_to_Joints3D(axis_angle1.numpy(), ret1.numpy())

        # 添加初始旋转和平移
        com_rec_left, com_rec_right = np.matmul(rot_from_x_to_0.reshape(1,1,3,3),joints.reshape(-1,22,3,1)),  np.matmul(rot_from_x_to_1.reshape(1,1,3,3),joints1.reshape(-1,22,3,1))
        com_rec_left, com_rec_right =  com_rec_left.reshape(-1,22,3) + dis_from_ori_to_0.reshape(1,1,3),  com_rec_right.reshape(-1,22,3) + dis_from_ori_to_1.reshape(1,1,3)
        print(joints.shape)

        # 平移可视化
        com_rec_left, com_rec_right = com_rec_left - com_rec_left[0][0], com_rec_right - com_rec_left[0][0]
        # print(com_rec_left)
        # print(com_rec_right)
        # assert 1==2
        # make_animation_matplot(joints.numpy(), joints1.numpy(),size=1.5)
        save_video = os.path.join(save_path, '{}.mp4'.format(name))
        make_animation_matplot(com_rec_left.numpy(), com_rec_right.numpy(),size=1.5,save_path=save_video)
        print("生成第{}个视频:".format(i))


def vis_pose_and_D(motion_0, motion_1, d_0, d_1, name, save_path):
    sample, sample1 = motion_0, motion_1
    canon0, canon1 = d_0, d_1
    sample, sample1 = torch.from_numpy(sample).unsqueeze(0), torch.from_numpy(sample1).unsqueeze(0)
    canon0, canon1 = torch.from_numpy(canon0), torch.from_numpy(canon1)
    canon0, canon1 = canon0, canon1

    rot_from_x_to_0, rot_from_x_to_1 = geometry.rotation_6d_to_matrix(canon0[:6]).numpy(), geometry.rotation_6d_to_matrix(canon1[:6]).numpy()
    dis_from_ori_to_0, dis_from_ori_to_1 = canon0[6:9].numpy(), canon1[6:9].numpy()
    
    print("0000::",sample.shape)
    # 1.3 motion数据恢复初始格式，反向归一化
    person_1, person_2 = sample.squeeze().cpu().reshape(-1,23,6), sample1.squeeze().cpu().reshape(-1,23,6)
    # person_1, person_2 = sample, sample1
    # mean_, std_ = torch.from_numpy(mean).cpu(), torch.from_numpy(std).cpu()
    # epsilon = 1e-6
    # # 找到所有为零的元素并替换为epsilon
    # std_[std_ == 0] = epsilon
    # # person_1, person_2 = person_1 * std_ + mean_,  person_2 * std_ + mean_ # [120,23,6]

    sample, ret = person_1[:,:22,:],  person_1[:,22,:3]
    sample1, ret1 = person_2[:,:22,:],  person_2[:,22,:3]

    # sample1 = sample.squeeze().cpu().permute(1,0).reshape(120,-1,6), sample1.squeeze().cpu().permute(1,0).reshape(120,-1,6)

    # 1.4 把motions转回成3Djoints坐标
    # ret, ret1 = sample[:,22,:3], sample1[:,22,:3]

    
    # ret, ret1 = ret*4, ret1*4
    motion, motion1 = sample[:,:22,:], sample1[:,:22,:]
    axis_angle, axis_angle1 = geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(motion)),  geometry.matrix_to_axis_angle(geometry.rotation_6d_to_matrix(motion1))

    joints, joints1 = Convert_Pose_to_Joints3D(axis_angle.numpy(), ret.numpy()), Convert_Pose_to_Joints3D(axis_angle1.numpy(), ret1.numpy())

    # 添加初始旋转和平移
    com_rec_left, com_rec_right = np.matmul(rot_from_x_to_0.reshape(1,1,3,3),joints.reshape(-1,22,3,1)),  np.matmul(rot_from_x_to_1.reshape(1,1,3,3),joints1.reshape(-1,22,3,1))
    com_rec_left, com_rec_right =  com_rec_left.reshape(-1,22,3) + dis_from_ori_to_0.reshape(1,1,3),  com_rec_right.reshape(-1,22,3) + dis_from_ori_to_1.reshape(1,1,3)
    print(joints.shape)

    # 平移可视化
    com_rec_left, com_rec_right = com_rec_left - com_rec_left[0][0], com_rec_right - com_rec_left[0][0]

    # make_animation_matplot(joints.numpy(), joints1.numpy(),size=1.5)
    save_video = os.path.join(save_path, '{}.mp4'.format(name))
    make_animation_matplot(com_rec_left.numpy(), com_rec_right.numpy(),size=1.5,save_path=save_video)
    np.save("../Camera/generated_data/joint_cvt/"+name+"_p0.npy",com_rec_left.numpy())
    np.save("../Camera/generated_data/joint_cvt/"+name+"_p1.npy",com_rec_right.numpy())
    print("生成第{}个视频:".format(name))

if __name__ == "__main__":
    folder_path = "../Camera/generated_data"
    for name in os.listdir(os.path.join(folder_path, 'new_joint_vecs/train')):# new_joint_vecs
        name = '0_p0.npy'
        motion_0 = np.load(os.path.join(folder_path, 'new_joint_vecs/train', name))
        motion_1 = np.load(os.path.join(folder_path, 'new_joint_vecs/train', name.replace('p0', 'p1')))

        d_0 = np.load(os.path.join(folder_path, 'canon_data/train', name))
        d_1 = np.load(os.path.join(folder_path, 'canon_data/train', name.replace('p0', 'p1')))
        print(motion_0.shape, d_0.shape)

    # 直接可视化npy文件斤
        vis_pose_and_D(motion_0, motion_1, d_0, d_1, name.split('.')[0], save_path="")

        assert 1==2
    

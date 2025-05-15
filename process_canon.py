import numpy as np
import json
import os,sys
import shutil
# from moviepy.editor import VideoFileClip, clips_array
import torch
# from utils.Convert_TRC_MOT import make_animation_matplot
# from utils.rotation_conversions import *
# from utils.humanml3d import Convert_Joints3D_to_Pose, BuildTrainData


def buildMultiVideos(file1,file2,file3=None):

    NameList = os.listdir(file1)
    for i in range(0, 87,2):
        # 视频文件路径列表
        c11, c12, c13 = os.path.join(file1,NameList[i]),  os.path.join(file1,NameList[i+1]),  os.path.join(file1,NameList[i+2])
        c21, c22, c23 = os.path.join(file2,NameList[i]),  os.path.join(file2,NameList[i+1]),  os.path.join(file2,NameList[i+2])
        # c31, c32, c33 = os.path.join(file3,NameList[i]),  os.path.join(file3,NameList[i+1]),  os.path.join(file3,NameList[i+2])
        print(c21)
        video_paths = [c11, c21, # c31,
                    c12, c22, # c32,
                    c13, c23]# c33

        # 加载视频并调整尺寸（如果需要）
        clips = [VideoFileClip(path).resize(width=320, height=240) for path in video_paths]

        # 创建九宫格布局
        final_clip = clips_array([[clips[0], clips[1]],# , clips[2]],
                                [clips[2], clips[3]],# , clips[5]],
                                [clips[4], clips[5]]])#, clips[8]]])

        # 输出最终视频
        final_clip.write_videofile(os.path.join("D:\Code\priorMDM-main\dataset\Experiments\\video\compare","four_grid_{}.mp4".format(i)))



def MoveTwoPersonCoords(left, right, mode='mid'):
    # left. right 的格式为[frames, joints, 3]
    # 以两个第一帧根节点的中心为坐标原点，把所有关节平移到原点两侧，方便后续计算根节点位移
    # root = (left[0,0,:] + right[0,0,:]) / 2
    if mode == 'mid':
        root = (left[0,0,:] +  right[0,0,:]) / 2
    else:
        root = left[0,0,:]# left[0,0,:]
    left = left - root
    right = right - root
    return left, right


def ReadTwoPersonData(file_path, move_to_ori=False):
    # 函数作用：读取一个数据文件路径，解析出left_person和right_person 的原始数据坐标
    # 返回两个人的坐标
    left_person_motions = []
    right_person_motions = []
    with open(file_path) as f:
        file = json.load(f)
        length = len(file['Feature'][0])
        for i in range(length):
            # print("==============当前是第{}个============".format(i))
            # print(file['Feature'][0][i])
            # print(file['Meta'])
            dic_i = json.loads(file['Feature'][0][i])
            # print(len(dic_i['Characters']['Left_Person']))
            left_person_motions.append(dic_i['Characters']['Left_Person'])
            right_person_motions.append(dic_i['Characters']['Right_Person'])
    f.close()
    left, right = MoveTwoPersonCoords(np.array(left_person_motions,dtype=np.float32), np.array(right_person_motions,dtype=np.float32),mode='mid')

    left[:,:,1], right[:,:,1] = left[:,:,1]*(-1), right[:,:,1]*(-1)
    # print(left.shape, right.shape)
    assert left.shape[1] == 24 and right.shape[1] == 24
    return left[:,:22,:], right[:,:22,:]

# 关节点平滑
def MotionFilter(left,right):
    from scipy.ndimage import gaussian_filter1d
    """
    对3D关节点坐标进行高斯平滑处理。

    :param data: 形状为[帧数, 关节点数, 3]的numpy数组，表示3D关节点坐标序列。
    :param sigma: 高斯核的标准差。
    :return: 高斯平滑后的数据。
    """
    sigma = 1.5
    smoothed_left = np.copy(left)
    smoothed_right = np.copy(right)
    num_frames, num_joints, _ = left.shape

    # 对每个关节点的每个坐标轴应用高斯平滑
    for joint in range(num_joints):
        for axis in range(3):
            smoothed_left[:, joint, axis] = gaussian_filter1d(left[:, joint, axis], sigma=sigma)
            smoothed_right[:, joint, axis] = gaussian_filter1d(right[:, joint, axis], sigma=sigma)

    return smoothed_left, smoothed_right

# 补帧，补到120
def AddOrDeleteClips(left, right):
    frames = left.shape[0]
    if frames < 120:
        flip_left, flip_right = np.flip(left,axis=0), np.flip(right,axis=0)
        added_frames = 120 - frames
        added_left, added_right = flip_left[1:added_frames+1], flip_right[1:added_frames+1]
        left, right = np.concatenate((left,added_left),axis=0), np.concatenate((right, added_right), axis=0)
    if frames > 120:
        left, right = left[:120], right[:120]

    return left, right


# 计算从A到B的旋转矩阵
def calculate_rotation_matrix(A, B):
    # 将向量A和B转换为单位向量
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)

    # 计算旋转轴（叉乘）
    v = np.cross(A, B)

    # 计算需要旋转的角度（点乘）
    cos_angle = np.dot(A, B)
    angle = np.arccos(cos_angle)

    # 罗德里格斯旋转公式组件
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    identity = np.eye(3)

    # 计算旋转矩阵
    rotation_matrix = identity + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return rotation_matrix

# 计算initial pose和real pose的相互旋转的矩阵
def ProcessInitPose(left_vector, right_vector):
    """
        left_vector, right_vector are shoulder_vector
    """
    referenced_axis = np.array([1,0,0])
    mat_from_x_to_left, mat_from_x_to_right = calculate_rotation_matrix(referenced_axis, left_vector), calculate_rotation_matrix(referenced_axis,right_vector)
    mat_from_left_to_x, mat_from_right_to_x = calculate_rotation_matrix(left_vector,referenced_axis), calculate_rotation_matrix(right_vector,referenced_axis)

    return mat_from_x_to_left, mat_from_x_to_right,mat_from_left_to_x,mat_from_right_to_x


## 随即旋转，创建一个绕y轴随即旋转的矩阵，扩充数据集，每个数据集扩充50份

def random_rotation_matrix_y():
    import random
    # 随机生成一个角度（弧度制）
    theta = random.uniform(0, 2*np.pi)

    # 构建绕Y轴的旋转矩阵
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_theta, 0, sin_theta],
                                [0, 1, 0],
                                [-sin_theta, 0, cos_theta]])

    return rotation_matrix

def random_rotate_joints(left_motions, right_motions):
    rot_mat = random_rotation_matrix_y()
    left_motions = np.matmul(rot_mat.reshape(1,1,3,3), left_motions.reshape(-1,22,3,1))
    right_motions = np.matmul(rot_mat.reshape(1,1,3,3), right_motions.reshape(-1,22,3,1))
    # make_animation_matplot(left_motions.reshape(-1,22,3), right_motions.reshape(-1,22,3), size=1.5)
    return left_motions.reshape(-1,22,3), right_motions.reshape(-1,22,3)

def ProcessTwoPerson_D(file_path):
    """
        1. 读取双人数据；
        2. 数据处理(归一化、增长到120帧)--根节点的位置很关键
        3. 数据格式转换(Joints 转 Poses)
    """
    #original_skeleton = np.load("D:\Code\motion-diffusion-model-main\dataset\Self_HumanML3D\joints\\000014.npy")
    # print(np.max(original_skeleton[:,:,0]),np.max(original_skeleton[:,:,1]),np.max(original_skeleton[:,:,2]))
    # print(np.min(original_skeleton[:,:,0]),np.min(original_skeleton[:,:,1]),np.min(original_skeleton[:,:,2]))

    # assert 1==2
    # save_dir = "D:\Code\priorMDM-main\dataset\Experiments\\completed_animations"
    # 拿到每个文件的名称
    count = 0
    jointsNameList = os.listdir(file_path)
    for name in jointsNameList:
        filename = os.path.join(file_path, name)
        print("文件名：",filename)
        # 1. 拿到平移后的两人joints
        left_joints, right_joints = ReadTwoPersonData(filename)
        left_joints, right_joints = left_joints / 800, right_joints / 800

        # 1.1 数据扩充
        for i in range(50):
            left_joints, right_joints = random_rotate_joints(left_joints, right_joints)

            # 2. 动作平滑
            smoothed_left, smoothed_right = MotionFilter(left_joints, right_joints)

            # 3.增删数据，把所有数据都填补到120帧。具体做法，少于120帧的，从最后一帧开始倒着填补，保证动作的连贯性；多余120帧，就截断
            com_left, com_right = AddOrDeleteClips(smoothed_left,smoothed_right)

            # 4.计算根节点相对于原点的位移
            ret_left, ret_right = com_left[0,0,:], com_right[0,0,:]
            #vret_left, ret_right = ret_left * 4, ret_right*4

            # 5.把com_left, com_right归置到各自原点
            com_ori_left, com_ori_right = com_left - com_left[0,0,:], com_right - com_right[0,0,:]

            # 6.设定一个轴，这里设为[1，0，0]
            left_v, right_v = com_ori_left[0,16,:] - com_ori_left[0,17,:], com_ori_right[0,16,:] - com_ori_right[0,17,:]
            mat_from_x_to_left, mat_from_x_to_right,mat_from_left_to_x,mat_from_right_to_x = ProcessInitPose(left_v, right_v)
            # 6.1 把动作统一到[1，0，0]方向
            rotated_left, rotated_right = np.matmul(mat_from_left_to_x.reshape(1,1,3,3), com_ori_left.reshape(-1,22,3,1)), np.matmul(mat_from_right_to_x.reshape(1,1,3,3), com_ori_right.reshape(-1,22,3,1))
            rotated_left, rotated_right = rotated_left.reshape(-1,22,3), rotated_right.reshape(-1,22,3)
            # make_animation_matplot(rotated_left, rotated_right,save_path=os.path.join("D:\Code\priorMDM-main\dataset\Experiments\\rotation_and_move\cur\\",name[:-5]+".mp4"),size=1.5)


            # 6.把D_添加回去
            com_rec_left, com_rec_right = np.matmul(mat_from_x_to_left.reshape(1,1,3,3),rotated_left.reshape(-1,22,3,1)),  np.matmul(mat_from_x_to_right.reshape(1,1,3,3),rotated_right.reshape(-1,22,3,1))
            com_rec_left, com_rec_right = com_rec_left.reshape(-1,22,3) + ret_left.reshape(1,1,3), com_rec_right.reshape(-1,22,3) + ret_right.reshape(1,1,3)

            make_animation_matplot(com_rec_left, com_rec_right,save_path=os.path.join("D:\Code\priorMDM-main\dataset\Experiments\\rotation_and_move\pre\\",name[:-5]+".mp4"),size=1.5)
            continue
            assert 1==2

            # 保存D的数据和处理后的人体动画数据
            #
            dict_person, dict_canon = {}, {}
            # 记录两个人的平移到空间原点之后的动作序列
            dict_person['left'], dict_person['right'] = rotated_left, rotated_right
            assert  dict_person['left'].shape[0] == 120 and dict_person['right'].shape[0] == 120
            rot6d_left, rot6d_right = matrix_to_rotation_6d(torch.from_numpy(mat_from_x_to_left)), matrix_to_rotation_6d(torch.from_numpy(mat_from_x_to_right))
            rot9d_left, rot9d_right = torch.cat([rot6d_left, torch.from_numpy(ret_left)],dim=0), torch.cat([rot6d_right, torch.from_numpy(ret_right)], dim=0)

            # 记录两个人各自的D
            dict_canon['left'], dict_canon['right'] = rot9d_left, rot9d_right
            np.save(os.path.join("D:\Code\priorMDM-main\dataset/Experiments/extend_joints",'{}.npy'.format(count)), dict_person)
            np.save(os.path.join("D:\Code\priorMDM-main\dataset/Experiments/extend_canon",'{}.npy'.format(count)), dict_canon)
            text_source_path = os.path.join("D:\Code\priorMDM-main\dataset/Experiments/texts/{}.txt".format(name.split('.')[0]))
            text_target_path = os.path.join("D:\Code\priorMDM-main\dataset/Experiments/extend_texts/{}.txt".format(count))
            shutil.copy(text_source_path, text_target_path)
            count += 1


###################################################################################################################
#########################    这里进行joints到pose的转换，转换完之后计算一下自己数据的均值和方差

def ReadTwoPersonData_from_Dict(joints_path):
    """
        joints文件保存在字典当中, 提出来
    """
    joints = np.load(joints_path,allow_pickle=True).item()
    # print(joints)
    left_joints, right_joints = joints['left'], joints['right']
    assert left_joints.shape[1] == right_joints.shape[1] == 22
    assert left_joints.shape[0] == right_joints.shape[0] == 120
    # print(left_joints.shape, right_joints.shape)
    return torch.from_numpy(left_joints).to(torch.float32),  torch.from_numpy(right_joints).to(torch.float32)

def ProcessJoints2Poses(joints_path):
    """

    """
    print("?>>>>>>")
    # 1.读取两个人的pose数据
    NameList = os.listdir(joints_path)
    for name in NameList:
        file_path = os.path.join(joints_path, name)
        # print(file_path)

        # 1.1 读取joints数据
        left_joints, right_joints = ReadTwoPersonData_from_Dict(file_path)

        # 1.2 转换成Pose
        pose_left, ret_left = Convert_Joints3D_to_Pose(left_joints.numpy())
        pose_right, ret_right = Convert_Joints3D_to_Pose(right_joints.numpy())

        # 1.3 转训练格式
        train_pose_left, train_pose_right = matrix_to_rotation_6d(axis_angle_to_matrix(pose_left)), matrix_to_rotation_6d(axis_angle_to_matrix(pose_right))
        BuildTrainData(train_pose_left,train_pose_right, ret_left, ret_right, save_path="D:\Code\priorMDM-main\dataset\Experiments\\extend_pose",file_name=name.split('.')[0])
    print('Finish Convertion !!')


#################################################################################################
def CalculateMeanAndStd(file_path):
    NameList = os.listdir(file_path)
    motions = []
    for i, name in enumerate(NameList):
        filename = os.path.join(file_path, name)
        poses = np.load(filename,allow_pickle=True).item()
        # print(joints)
        left, right = poses['left'], poses['right']
        left, right = torch.from_numpy(left).to(torch.float32), torch.from_numpy(right).to(torch.float32)
        has_left_nan, has_right_nan = torch.any(torch.isnan(left)), torch.any(torch.isnan(left))
        if has_left_nan or has_right_nan:
            print(name)
            continue
            print(i)
        motions.append(left.numpy())
        motions.append(right.numpy())

    mean = np.mean(np.array(motions),axis=0)
    std = np.std(np.array(motions), axis=0)
    # print(mean.shape)
    # print(std.shape)
    has_nan_mean = np.any(np.isnan(mean))
    has_nan_std = np.any(np.isnan(std))
    print("结果包含NaN值：", has_nan_mean, has_nan_std)

    np.save("D:\Code\priorMDM-main\dataset\Experiments\Mean.npy", mean)
    np.save("D:\Code\priorMDM-main\dataset\Experiments\Std.npy", std)

def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

# if __name__ == '__main__':
#     file_path = "D:\Code\priorMDM-main\dataset\Experiments\\new_name_joints"
#     train_joints_path = "D:\Code\priorMDM-main\dataset\Experiments\extend_joints"
#     train_poses_path = "D:\Code\priorMDM-main\dataset\Experiments\\extend_pose"

#     # 1.把原始3D坐标的数据处理成相对于空间原点后，计算D，并平移到原点
#     ProcessTwoPerson_D(file_path=file_path)

#     # 2.把train_joints中的坐标数据，转换成训练需要的轴角+位移数据格式
#     # ProcessJoints2Poses(train_joints_path)

#     # 3.计算所有数据的Mean和std
#     # CalculateMeanAndStd(train_poses_path)


#     f1 = "D:\Code\priorMDM-main\\temporary_folder\\test_my_MDM\samples\\video"
#     f2 = "D:\Code\priorMDM-main\\temporary_folder\\test_ini_MDM\samples\\video"
#     # f3 = "D:\Code\priorMDM-main\dataset\Experiments\\rotation_and_move\pre"
#     # buildMultiVideos(f1,f2)

import torch
# from lie.lie_util import *
from torch import nn


import torch

HAT_INV_SKEW_SYMMETRIC_TOL = 1e-5


def allclose(mat1, mat2, tol=1e-6):
    '''
    check is all elements of two tensors are close within some tolerance.

    Either tensor can be replaced by a scalar
    '''
    return isclose(mat1, mat2, tol).all()


def isclose(mat1, mat2, tol=1e-6):
    """
    check element-wise if two tensors are close within some tolerance.

    Either tensot can be replaced by a scalar
    """
    return (mat1 - mat2).abs_().lt(tol)

def outer(vecs1, vecs2):
    """Return the N x D x D outer products of a N x D batch of vectors,
    or return the D x D outer product of two D-dimensional vectors.
    """
    # Default batch size is 1
    if vecs1.dim() < 2:
        vecs1 = vecs1.unsqueeze(dim=0)

    if vecs2.dim() < 2:
        vecs2 = vecs2.unsqueeze(dim=0)

    if vecs1.shape[0] != vecs2.shape[0]:
        raise ValueError("Got inconsistent batch sizes {} and {}".format(
            vecs1.shape[0], vecs2.shape[0]))

    return torch.bmm(vecs1.unsqueeze(dim=2),
                     vecs2.unsqueeze(dim=2).transpose(2, 1)).squeeze_()


def trace(mat):
    """Return the N traces of a batch of N square matrices,
    or return the trace of a square matrix."""
    # Default batch size is 1
    if mat.dim() < 3:
        mat = mat.unsqueeze(dim=0)

    # Element-wise multiply by identity and take the sum
    tr = (torch.eye(mat.shape[1], dtype=mat.dtype) * mat).sum(dim=1).sum(dim=1)

    return tr.view(mat.shape[0])


def matR_log_map(R, eps: float = 1e-4, cos_angle: bool = False):
    """
    Returns the axis angle aka lie algebra parameters from a rotation matrix(SO3)
    """
    N, dim1, dim2 = R.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    rot_trace = trace(R)

    if((rot_trace < -1.0 - eps) + (rot_trace > 3.0 + eps)).any():
        raise ValueError(
            "A matrix has trace outside valid range [-1-eps, 3+eps]."
        )

    # clamp to valid range
    rot_trace = torch.clamp(rot_trace, -1.0, 3.0)

    # phi ... rotation angle
    phi = 0.5 * (rot_trace - 1.0)
    phi = phi.acos()

    phi_valid = torch.clamp(phi.abs(), eps) * phi.sign()

    log_rot_hat = (phi_valid / (2.0 * phi_valid.sin()))[:, None, None] * (
        R - R.permute(0, 2, 1)
    )
    log_rot = hat_inv(log_rot_hat)
    return log_rot


def lie_u_v(u, v, eps: float = 1e-4):
    """
    find the axis angle parameters to rotate unit vector u onto unit vector v
    which is also the lie algebra parameters of corresponding SO3
    """
    w = torch.cross(u, v, dim=1)

    w_norm = torch.norm(w, p=2, dim=1)
    w_norm = torch.clamp(w_norm, eps)
    A = w / w_norm[:, None] * torch.mul(u, v).sum(dim=1).acos()[:, None]
    return A


def lie_exp_map(log_rot, eps: float = 1e-4):
    """
    Convert the lie algebra parameters to rotation matrices
    """
    _, dim = log_rot.shape
    if dim != 3:
        raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    R = (
        fac1[:, None, None] * skews
        + fac2[:, None, None] * torch.bmm(skews, skews)
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )
    # print(R.shape)
    return R


def hat(v):
    """
    compute the skew-symmetric matrices with a batch of 3d vectors.
    """
    N, dim = v.shape
    if dim != 3:
        raise ValueError("Input vectors have to be 3-dimensional.")
    h = v.new_zeros(N, 3, 3)
    x, y, z = v.unbind(1)
    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h


def hat_inv(h):
    """
    compute the 3d-vectors with a batch of 3x3 skew-symmetric matrics.
    """
    N, dim1, dim2 = h.shape
    if dim1 != 3 or dim2 != 3:
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    ss_diff = (h + h.permute(0, 2, 1)).abs().max()
    if float(ss_diff) > HAT_INV_SKEW_SYMMETRIC_TOL:
        raise ValueError("One of input matrices not skew-symmetric.")

    x = h[:, 2, 1]
    y = h[:, 0, 2]
    z = h[:, 1, 0]

    v = torch.stack((x, y, z), dim=1)

    return v
############################################################################################################

class LieSkeleton(object):
    def __init__(self, raw_translation, kinematic_tree, tensor):
        super(LieSkeleton, self).__init__()
        self.tensor = tensor
        # print(self.tensor)
        self._raw_translation = self.tensor(raw_translation.shape).copy_(raw_translation).detach()
        self._kinematic_tree = kinematic_tree
        self._translation = None
        self._parents = [0] * len(self._raw_translation)
        self._parents[0] = -1
        for chain in self._kinematic_tree:
            for j in range(1, len(chain)):
                self._parents[chain[j]] = chain[j-1]

    def njoints(self):
        return len(self._raw_translation)

    def raw_translation(self):
        return self._raw_translation

    def kinematic_tree(self):
        return self._kinematic_tree

    def parents(self):
        return self._translation

    def get_translation_joints(self, joints):
        # joints/offsets (batch_size, joints_num, 3)
        # print(self._raw_translation.shape)
        _translation = self._raw_translation.clone().detach()
        _translation = _translation.expand(joints.shape[0], -1, -1).clone()
        #print(_translation.shape)
        #print(self._raw_translation.shape)
        for i in range(1, self._raw_translation.shape[0]):
            _translation[:, i, :] = torch.norm(joints[:, i, :] - joints[:, self._parents[i], :], p=2, dim=1)[:, None] * \
                                     _translation[:, i, :]
        self._translation = _translation
        return _translation

    def get_translation_bone(self, bonelengths):
        # bonelength (batch_size, joints_num - 1)
        # offsets (batch_size, joints_num, 3)
        self._translation = self._raw_translation.clone().detach().expand(bonelengths.size(0), -1, -1).clone().to(bonelengths.device)
        self._translation[:, 1:, :] = bonelengths * self._translation[:, 1:, :]

    def inverse_kinemetics(self, joints):
        # joints (batch_size, joints_num, 3)
        # lie_params (batch_size, joints_num, 3)
        lie_params = self.tensor(joints.shape).fill_(0)
        # root_matR (batch_size, 3, 3)
        root_matR = torch.eye(3, dtype=joints.dtype).expand((joints.shape[0], -1, -1)).clone().detach().to(joints.device)
        for chain in self._kinematic_tree:
            R = root_matR
            for j in range(len(chain) - 1):
                # (batch, 3)
                u = self._raw_translation[chain[j + 1]].expand(joints.shape[0], -1).clone().detach().to(joints.device)
                # (batch, 3)
                v = joints[:, chain[j+1], :] - joints[:, chain[j], :]
                # (batch, 3)
                v = v / torch.norm(v, p=2, dim=1)[:, None]
                # (batch, 3, 3)
                R_local = torch.matmul(R.transpose(1, 2), lie_exp_map(lie_u_v(u, v)))
                # print("R_local shape:" + str(R_local.shape))
                # print(R_local)
                # t = lie_u_v(u, v)
                lie_params[:, chain[j + 1], :] = matR_log_map(R_local)

                R = torch.matmul(R, R_local)

        return lie_params

    def forward_kinematics(self, lie_params, joints, root_translation, do_root_R = False, scale_inds=None):
        # lie_params (batch_size, joints_num, 3) lie_params[:, 0, :] is not used
        # joints (batch_size, joints_num, 3)
        # root_translation (batch_size, 3)
        # translation_mat (batch_size, joints_num, 3)
        translation_mat = self.get_translation_joints(joints)
        if scale_inds is not None:
            translation_mat[:, scale_inds, :] *= 1.25
        joints = self.tensor(lie_params.size()).fill_(0)
        joints[:, 0] = root_translation
        for chain in self._kinematic_tree:
            # if do_root_R is true, root coordinate system has rotation angulers
            # Plus, for chain not containing root(e.g arms), we use root rotation as the rotation
            # of joints near neck(i.e. beginning of this chain).
            if do_root_R:
                matR = lie_exp_map(lie_params[:, 0, :])
            # Or, root rotation matrix is identity matrix, which means no rotation at global coordinate system
            else:
                matR = torch.eye(3, dtype=joints.dtype).expand((joints.shape[0], -1, -1)).clone().detach().to(joints.device)
            for i in range(1, len(chain)):
                matR = torch.matmul(matR, lie_exp_map(lie_params[:, chain[i], :]))
                translation_vec = translation_mat[:, chain[i], :].unsqueeze_(-1)
                joints[:, chain[i], :] = torch.matmul(matR, translation_vec).squeeze_()\
                                         + joints[:, chain[i-1], :]
        return joints
def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))
def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))
# Define a kinematic tree for the skeletal struture
humanact12_kinematic_chain = [[0, 1, 4, 7, 10], [0, 2, 5, 8, 11], [0, 3, 6, 9, 12, 15], [9, 13, 16, 18, 20], [9, 14, 17, 19, 21]]
humanact12_raw_offsets = np.array([[0,0,0],
                               [1,0,0],
                               [-1,0,0],
                               [0,1,0],
                               [0,-1,0],
                               [0,-1,0],
                               [0,1,0],
                               [0,-1,0],
                               [0,-1,0],
                               [0,1,0],
                               [0,0,1],
                               [0,0,1],
                               [0,1,0],
                               [1,0,0],
                               [-1,0,0],
                               [0,0,1],
                               [0,-1,0],
                               [0,-1,0],
                               [0,-1,0],
                               [0,-1,0],
                               [0,-1,0],
                               [0,-1,0]])

def Convert_Joints3D_to_Pose(Joints3D, mode='humanact12'):
    # index = 887
    # x_Joints = np.array(data['joints3D'][index])
    x_Joints = Joints3D
    # print("Verify joints:",x_Joints[1])
    offet_mat =  np.tile(x_Joints[0, 0], (x_Joints.shape[1], 1))
    joints3D = x_Joints - offet_mat

    ret = joints3D[:,0,:] # 每一帧根节点的偏移
    # print(ret)
    # make_animation_matplot(joints3D)
    # x_pose = np.array(data['poses'][index]).reshape(-1,24,3)
    if mode == 'smpl':
        raw_offsets = torch.from_numpy(smpl_skeleton).to(torch.float32)
    else:
        raw_offsets = torch.from_numpy(humanact12_raw_offsets).to(torch.float32)
    # 定义骨架类

    lie_skeleton = LieSkeleton(raw_offsets, humanact12_kinematic_chain, torch.FloatTensor)

    pose_mat = lie_skeleton.inverse_kinemetics(torch.from_numpy(joints3D))# .numpy()
    return pose_mat, ret


def BuildTrainData(left, right, ret_left, ret_right, save_path=None,file_name=None):
    save_dict = {}
    padded_left, padded_right = torch.zeros((ret_left.shape[0], 6)), torch.zeros((ret_right.shape[0], 6))
    padded_left[:, :3], padded_right[:, :3] = torch.from_numpy(ret_left), torch.from_numpy(ret_right)
    padded_left, padded_right = padded_left.unsqueeze(1), padded_right.unsqueeze(1)
    left_data, right_data = torch.cat((left, padded_left), 1), torch.cat((right, padded_right), 1),
    assert left_data.shape == right_data.shape
    # print(left_data.shape)
    save_dict['left'] = left_data.numpy()
    save_dict['right'] = right_data.numpy()
    # assert save_dict['left'].shape[0] == save_dict['right'].shape[0] == 120
    assert save_dict['left'].shape[1] == save_dict['right'].shape[1] == 23
    assert save_dict['left'].shape[2] == save_dict['right'].shape[2] == 6

    return left_data.numpy(), right_data.numpy()

import numpy as np
import os
from matplotlib.animation import FuncAnimation
import copy
import matplotlib.pyplot as plt



lines = [[0,1],[0,2],[2,5],[5,8],[8,11],[1,4],[4,7],[7,10],[0,3],[3,6],[6,9],[9,13],[9,14],[15,12],[12,16],[12,17],[17,19],[19,21],[16,18],[18,20]]


def make_animation_matplot(data1, data2=None,save_path=None, size=1):
    frames = len(data1)
    # 创建图形和3D坐标轴
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置坐标轴标签
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    RADIUS = size # space around the subject
    # xroot, yroot, zroot = data1[0,0,0], data1[0,0,1], data1[0,0,2] #hip的位置
    xroot, yroot, zroot = (data1[0,0,0] + data2[0,0,0]) / 2, (data1[0,0,1] + data2[0,0,1])/2, (data1[0,0,2]+data2[0,0,2])/2 #hip的位置

    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

    # 定义连接点的序列，这取决于骨架的结构
    # 以下是一个假设的例子
    connections = lines

    c1 = copy.deepcopy(data1)# , copy.deepcopy(data2)
    c2 = copy.deepcopy(data2)

    data1[:,:,0], data1[:,:,1], data1[:,:,2] = c1[:,:,0], c1[:,:,2], c1[:,:,1]
    data2[:,:,0], data2[:,:,1], data2[:,:,2] = c2[:,:,0], c2[:,:,2], c2[:,:,1]

    # data1 *= -1
    # data2 *= -1

    # 初始化两个骨架的散点图和线段
    scat1 = ax.scatter(data1[0, :, 0], data1[0, :, 1], data1[0, :, 2], color='blue')
    scat2 = ax.scatter(data2[0, :, 0], data2[0, :, 1], data2[0, :, 2], color='red')
    lines1 = [ax.plot([data1[0, start, 0], data1[0, end, 0]],
                    [data1[0, start, 1], data1[0, end, 1]],
                    [data1[0, start, 2], data1[0, end, 2]], color='blue')[0] for start, end in connections]
    lines2 = [ax.plot([data2[0, start, 0], data2[0, end, 0]],
                    [data2[0, start, 1], data2[0, end, 1]],
                    [data2[0, start, 2], data2[0, end, 2]], color='red')[0] for start, end in connections]

    # 更新函数，用于动画
    def update(frame):
        scat1._offsets3d = (data1[frame, :, 0], data1[frame, :, 1], data1[frame, :, 2])
        # scat2._offsets3d = (data2[frame, :, 0], data2[frame, :, 1], data2[frame, :, 2])
        for line, (start, end) in zip(lines1, connections):
            line.set_data([data1[frame, start, 0], data1[frame, end, 0]],
                        [data1[frame, start, 1], data1[frame, end, 1]])
            line.set_3d_properties([data1[frame, start, 2], data1[frame, end, 2]])

        for line, (start, end) in zip(lines2, connections):
            line.set_data([data2[frame, start, 0], data2[frame, end, 0]],
                        [data2[frame, start, 1], data2[frame, end, 1]])
            line.set_3d_properties([data2[frame, start, 2], data2[frame, end, 2]])

        return scat1, *lines1, *lines2

        # return scat1, *lines1#, *lines2

        # 创建动画
    ani = FuncAnimation(fig, update, frames=frames, interval=100, blit=False)
    plt.show()
    ani.save(save_path,writer='ffmpeg', fps=20)

###########################################################################################
# file1 = "/content/drive/MyDrive/motions_processed/person1/51.npy"
# file2 = "/content/drive/MyDrive/motions_processed/person2/51.npy"

def Build_D(p1, p2):
  # motion1, motion2 = np.load(p1),np.load(p2)
  motion1, motion2 = p1, p2
  rot_R = np.asarray([[1,0,0],[0,0,1],[0,1,0]])
  p1, p2 = motion1[:,:22*3].reshape(-1,22,3), motion2[:,:22*3].reshape(-1,22,3)
  p1_, p2_ = np.matmul(p1, rot_R), np.matmul(p2, rot_R)
  # make_animation_matplot(p1_, p2_,save_path='/content/test.mp4')
  # assert 1==2
  # 处理双人数据
  print(p1.shape)

  # 补帧
  if len(p1_) < 120:
    com_left, com_right = AddOrDeleteClips(p1_,p2_)
  else:
    com_left, com_right = p1_[:120], p2_[:120]
  # 计算根节点相对于原点的位移
  ret_left, ret_right = com_left[0,0,:], com_right[0,0,:]
  # 把com_left, com_right归置到各自原点
  com_ori_left, com_ori_right = com_left - com_left[0,0,:], com_right - com_right[0,0,:]
  # make_animation_matplot(com_ori_left, com_ori_right,save_path='/content/test.mp4')
  # 设定一个轴，这里设为[1，0，0]
  left_v, right_v = com_ori_left[0,16,:] - com_ori_left[0,17,:], com_ori_right[0,16,:] - com_ori_right[0,17,:]
  mat_from_x_to_left, mat_from_x_to_right,mat_from_left_to_x,mat_from_right_to_x = ProcessInitPose(left_v, right_v)
  #rint(mat_from_x_to_left)
  # 把动作统一到[1，0，0]方向
  rotated_left, rotated_right = np.matmul(mat_from_left_to_x.reshape(1,1,3,3), com_ori_left.reshape(-1,22,3,1)), np.matmul(mat_from_right_to_x.reshape(1,1,3,3), com_ori_right.reshape(-1,22,3,1))
  rotated_left, rotated_right = rotated_left.reshape(-1,22,3), rotated_right.reshape(-1,22,3)

  ###  构建motion用于训练数据
  # 转换成Pose
  pose_left, r_left = Convert_Joints3D_to_Pose(rotated_left.astype(np.float32))
  pose_right, r_right = Convert_Joints3D_to_Pose(rotated_right.astype(np.float32))
  # 转训练格式
  train_pose_left, train_pose_right = matrix_to_rotation_6d(axis_angle_to_matrix(pose_left)), matrix_to_rotation_6d(axis_angle_to_matrix(pose_right))

  final_l, final_r = BuildTrainData(train_pose_left, train_pose_right,r_left, r_right)

  #return final_l, final_r

#   """
  # make_animation_matplot(rotated_left, rotated_right,save_path='/content/test.mp4')
  # 把D_添加回去
  com_rec_left, com_rec_right = np.matmul(mat_from_x_to_left.reshape(1,1,3,3),rotated_left.reshape(-1,22,3,1)),  np.matmul(mat_from_x_to_right.reshape(1,1,3,3),rotated_right.reshape(-1,22,3,1))
  com_rec_left, com_rec_right = com_rec_left.reshape(-1,22,3) + ret_left.reshape(1,1,3), com_rec_right.reshape(-1,22,3) + ret_right.reshape(1,1,3)
  # make_animation_matplot(com_rec_left, com_rec_right,save_path='/content/test.mp4')
  # 构造D
  rot6d_left, rot6d_right = matrix_to_rotation_6d(torch.from_numpy(mat_from_x_to_left)), matrix_to_rotation_6d(torch.from_numpy(mat_from_x_to_right))
  rot9d_left, rot9d_right = torch.cat([rot6d_left, torch.from_numpy(ret_left)],dim=0), torch.cat([rot6d_right, torch.from_numpy(ret_right)], dim=0)

  return rot9d_left, rot9d_right
  # """

def extract_smpl_from_smplx(smplx_file):
  pass

def split_two_person(folder_path):
  namelist = os.listdir(folder_path)
  for name in namelist:
    key_name = name[:-7]
    p1, p2 = np.load(os.path.join(folder_path,key_name+"_00.npy")), np.load(os.path.join(folder_path,key_name+"_01.npy"))

if __name__ == '__main__':
    folder_path = "../priorMD/dataset/Siggraph_data (2)/Siggraph_data/JOINT/"
    namelist = os.listdir("../priorMD/dataset/Siggraph_data (2)/Siggraph_data/JOINT/person1")
    #save_path = 'Dataset/canon_data'
    save_path = '../priorMD/dataset/Siggraph_data (2)/Siggraph_data/processed_data/canon_data'

    ttt = []
    for name in namelist:
        p1, p2 = np.load(os.path.join(folder_path,'person1',name)), np.load(os.path.join(folder_path,'person2',name))
        d1, d2 = Build_D(p1, p2)
        ttt.append(d1.shape[0])
        # assert d1.shape == d2.shape == (138,)
        s1 = os.path.join(save_path, 'train', '{}_p0.npy'.format(name.split('.')[0]))
        s2 = os.path.join(save_path, 'train', '{}_p1.npy'.format(name.split('.')[0]))

        np.save(s1, d1)
        np.save(s2, d2)
    print("Finish Process Canon!!")
    print(ttt)




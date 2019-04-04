import os
import pickle
import random
import numpy as np
from mxnet import ndarray as nd
from .imdb import IMDB
# from utils import LOG
from config import config

s_hm36_subject_num = 7
HM_subject_idx = [ 1, 5, 6, 7, 8, 9, 11 ]
HM_subject_idx_inv = [ -1, 0, -1, -1, -1, 1, 2, 3, 4, 5, -1, 6 ]

s_hm36_act_num = 15
HM_act_idx = [ 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 ]
HM_act_idx_inv = [ -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ]

s_hm36_subact_num = 2
HM_subact_idx = [ 1, 2 ]
HM_subact_idx_inv = [ -1, 0, 1 ]

s_hm36_camera_num = 4
HM_camera_idx = [ 1, 2, 3, 4 ]
HM_camera_idx_inv = [ -1, 0, 1, 2, 3 ]

# 17 joints of Human3.6M:
# 'root', 'Rleg0', 'Rleg1', 'Rleg2', 'Lleg0', 'Lleg1', 'Lleg2', 'Spine', 'Thorax', 'Neck/Nose', 'head', 'Larm0', 'Larm1', 'Larm2', 'Rarm0', 'Rarm1', 'Rarm2'
# 'root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'Spine', 'Thorax', 'Neck/Nose', 'head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist'

# 18 joints with Thorax:
# 'root', 'Rleg0', 'Rleg1', 'Rleg2', 'Lleg0', 'Lleg1', 'Lleg2', 'torso', 'neck', 'nose', 'head', 'Larm0', 'Larm1', 'Larm2', 'Rarm0', 'Rarm1', 'Rarm2', 'Thorax'
# 'root', 'RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'torso', 'neck', 'nose', 'head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist, 'Thorax''
# 0       1       2        3         4       5        6         7        8       9       10      11           12        13        14           15        16       17
# [ 0,      0,      1,       2,        0,      4,       5,        0,      17,      17,      8,     17,           11,        12,       17,          14,       15,      0]

# 16 joints of MPII
# 0-R_Ankle, 1-R_Knee, 2-R_Hip, 3-L_Hip, 4-L_Knee, 5-L_Ankle, 6-Pelvis, 7-Thorax,
# 8-Neck, 9-Head, 10-R_Wrist, 11-R_Elbow, 12-R_Shoulder, 13-L_Shoulder, 14-L_Elbow, 15-L_Wrist

JntName = ['RHip', 'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle', 'Spine', 'Thorax',
           'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']

s_org_36_jt_num = 32
s_36_root_jt_idx = 0
s_36_lsh_jt_idx = 11
s_36_rsh_jt_idx = 14
s_36_jt_num = 17
# jt_list = [1, 2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]
# s_36_flip_pairs = np.array([[1, 4], [2, 5], [3, 6], [14, 11], [15, 12], [16, 13]], dtype=np.int)
# s_36_parent_ids = np.array([0, 0, 1, 2, 0, 4, 5, 0, 17, 17, 8, 17, 11, 12, 17, 14, 15, 0], dtype=np.int)
s_mpii_2_hm36_jt = [6, 2, 1, 0, 3, 4, 5, -1, 8, -1, 9, 13, 14, 15, 12, 11, 10, 7]
s_hm36_2_mpii_jt = [3, 2, 1, 4, 5, 6, 0, 17, 8, 10, 16, 15, 14, 11, 12, 13]
s_rect_3d_size = 2000

def parsing_hm36_gt_file(gt_file):
    '''return keypoints, trans, rot, fl, c_p, k_p, p_p'''
    keypoints = []
    with open(gt_file, 'r') as f:
        content = f.readlines()
        image_num = int(float(content[0]))
        img_width = content[1].split(' ')[1]
        img_height = content[1].split(' ')[2]
        rot = content[2].split(' ')[1:10]
        trans = content[3].split(' ')[1:4]
        fl = content[4].split(' ')[1:3]  #focal length
        c_p = content[5].split(' ')[1:3] #camera params
        k_p = content[6].split(' ')[1:4] #radial distortion
        p_p = content[7].split(' ')[1:3] #tangent distortion   
        jt_list = content[8].split(' ')[1:18]
        for i in range(0, image_num):
            keypoints.append(content[9 + i].split(' ')[1:97])

    keypoints = np.asarray([[float(y) for y in x] for x in keypoints])
    keypoints = keypoints.reshape(keypoints.shape[0], int(keypoints.shape[1] / 3), 3)
    trans = np.asarray([float(y) for y in trans])
    jt_list = np.asarray([int(y) for y in jt_list])   # fetch necessary 17 joints index
    keypoints = keypoints[:, jt_list - 1, :] 

    rot = np.asarray([float(y) for y in rot]).reshape((3,3))
    rot = np.transpose(rot)
    fl  = np.asarray([float(y) for y in fl])
    c_p = np.asarray([float(y) for y in c_p])
    k_p = np.asarray([float(y) for y in k_p])
    p_p = np.asarray([float(y) for y in p_p])

    return keypoints, trans, rot, fl, c_p, k_p, p_p


def CamProj(x, y, z, fx, fy, u, v):
    cam_x = x / z * fx
    cam_x = cam_x + u
    cam_y = y / z * fy
    cam_y = cam_y + v
    return cam_x, cam_y


def CamBackProj(cam_x, cam_y, depth, fx, fy, u, v):
    x = (cam_x - u) / fx * depth
    y = (cam_y - v) / fy * depth
    z = depth
    return x, y, z


class hm36(IMDB):
    def __init__(self, image_set, root_path, data_path, result_path=None):
        super(hm36, self).__init__('HM36', image_set, root_path, data_path, result_path)
        self.joint_num = s_36_jt_num


    def _H36FolderName(self, subject_id, act_id, subact_id, camera_id):
        return "s_%02d_act_%02d_subact_%02d_ca_%02d" % (HM_subject_idx[subject_id], HM_act_idx[act_id], HM_subact_idx[subact_id], HM_camera_idx[camera_id])


    def _H36ImageName(self, folder_name, frame_id):
        return "%s_%06d.jpg" % (folder_name, frame_id + 1)


    def _AllHuman36Folders(self, subject_list_):
        subject_list = subject_list_[:]
        if len(subject_list) == 0:
            for i in range(0, s_hm36_subject_num):
                subject_list.append(i)
        folders = []
        for i in range(0, len(subject_list)):
            for j in range(0, s_hm36_act_num):
                for m in range(0, s_hm36_subact_num):
                    for n in range(0, s_hm36_camera_num):
                        folders.append(self._H36FolderName(subject_list[i], j, m, n))
        return folders


    def get_meanstd(self, db, logger):
        # use the training data meanstd also for valid/test data
        cache_file = os.path.join(self.cache_path, 'noise_sigma%d_'%config.DATASET.sigma +
                                  'HM36_meanstd_sample%02d'%(self.sample_step) + '.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                meanstd = pickle.load(fid)
            logger.info('{} meanstd loaded from {}'.format(self.name, cache_file))
            return meanstd['mean2d'], meanstd['std2d'], meanstd['mean3d'], meanstd['std3d']
        
        total2D = []
        total3D = []
        for sample in db:
            total2D.append(sample['joints_2d'].flatten())
            total3D.append(sample['joints_3d'].flatten())

        total2D = np.asarray(total2D)
        total3D = np.asarray(total3D)
        
        mean2d = total2D.mean(axis=0)
        std2d  = total2D.std(axis=0)
        mean3d = total3D.mean(axis=0)
        std3d  = total3D.std(axis=0)

        with open(cache_file, 'wb') as fid:
            pickle.dump({'mean2d': mean2d,'std2d': std2d,'mean3d': mean3d,'std3d': std3d}, fid, pickle.HIGHEST_PROTOCOL)
        logger.info('{} meanstd are written into {}'.format(self.name, cache_file))            
        return mean2d, std2d, mean3d, std3d


    def gt_db_actions(self, action, logger):
        '''get 3D ground truth for this action from database'''
        if not logger:
            assert False, 'require a logger'

        if self.image_set == 'test':
            self.sample_step = 1
            self.folder_start = 0
            self.folder_end = 240
            folders = self._AllHuman36Folders([5, 6])
        else:
            logger.info('Error!!!!!!!!! Unknown hm36 subset!')
            assert 0

        # if cache exist, load from it, 
        # here same sample is not needed for test, cause no need to repeat test data,
        # only to repeat training data
        cache_file = os.path.join(self.cache_path, 'noise_sigma%d_'%config.DATASET.sigma +
                                  self.name + '_sample%02d'%(self.sample_step) + '_act' + action + '.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                db = pickle.load(fid)
            logger.info('{} gt db loaded from {}, {} samples are loaded'.format(self.name, cache_file, len(db)))
            self.num_images = len(db)
            return db
        
        # if cache doesn't exist, generate and save
        gt_db = []
        for n_folder in range(self.folder_start, self.folder_end):
            logger.info('Loading folder ', n_folder+1, ' in ', len(folders))

            # load ground truth
            txtPath = os.path.join(self.data_path, "annot", folders[n_folder], 'matlab_meta.txt')
            actionNum = txtPath.split('/')[-2].split('_')[3]
            if actionNum != action:
                continue
            keypoints, trans, rot, fl, c_p, k_p, p_p = parsing_hm36_gt_file(txtPath)

            # sample redundant video sequence
            img_index = np.arange(0, keypoints.shape[0], step=self.sample_step)
            for n_img_ in range(0, img_index.shape[0]):
                '''Fetch image and labels'''
                n_img = img_index[n_img_]
                assert keypoints.shape[1] == self.joint_num

                # project to image coordinate, and get 2D&3D label
                pt_2d = np.zeros((self.joint_num, 2), dtype=np.float)
                pt_3d = np.zeros((self.joint_num, 3), dtype=np.float)
                for n_jt in range(0, self.joint_num):
                    # tranform 3d joints from world space to camera space
                    pt_3d[n_jt] = np.dot(rot, keypoints[n_img, n_jt] - trans)  
                    # undistortion, wait to fill

                    # project 3d joints into image 2d joints
                    pt_2d[n_jt, 0], pt_2d[n_jt, 1] = CamProj(pt_3d[n_jt, 0], pt_3d[n_jt, 1], pt_3d[n_jt, 2], fl[0], fl[1], c_p[0], c_p[1])
                
                # substract root and remove
                pelvis3d = pt_3d[s_36_root_jt_idx]
                pt_3d = pt_3d - pelvis3d
                pt_3d = np.delete(pt_3d, 0, axis=0) #(16,3)

                # remove neck/nose
                pt_2d = np.delete(pt_2d, 9, axis=0) #(16,2)

                #if sigma==5 nothing changes
                noise = np.random.normal(0,config.DATASET.sigma,32).reshape((16,2))
                pt_2d = pt_2d + noise
                    
                gt_db.append({
                    'joints_2d': pt_2d, # [org_img_x, org_img_y]
                    'joints_3d': pt_3d, # [X, Y, Z] in camera coordinate
                    'folder': n_folder
                })

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_db, fid, pickle.HIGHEST_PROTOCOL)
        logger.info('{} samples ared wrote {}'.format(len(gt_db), cache_file))
        return gt_db     


    def gt_db(self, logger):
        '''get 3D grount truth from database'''
        if not logger:
            assert False, 'require a logger'

        # sampling to reduce redundance
        if self.image_set == 'train':          
            self.sample_step = 1
            self.folder_start = 0
            self.folder_end = 600
            folders = self._AllHuman36Folders([0, 1, 2, 3, 4])
        elif self.image_set == 'valid':
            self.sample_step = 1
            self.folder_start = 0
            self.folder_end = 240
            folders = self._AllHuman36Folders([5, 6])
        else:
            logger.info('Error!!!!!!!!! Unknown hm36 subset!')
            assert 0

        # if cache exist, load from it
        cache_file = os.path.join(self.cache_path, 'noise_sigma%d_'%config.DATASET.sigma +
                                  self.name + '_sample%02d'%(self.sample_step) + '.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                db = pickle.load(fid)
            logger.info('{} gt db loaded from {}, {} samples are loaded'.format(self.name, cache_file, len(db)))
            self.num_images = len(db)
            return db
        
        # if cache doesn't exist, generate and save
        gt_db = []
        for n_folder in range(self.folder_start, self.folder_end):
            logger.info('Loading folder ', n_folder+1, ' in ', len(folders))

            # load ground truth
            txtPath = os.path.join(self.data_path, "annot", folders[n_folder], 'matlab_meta.txt')
            # actionNum = int(txtPath.split('/')[-2].split('_')[3])
            keypoints, trans, rot, fl, c_p, k_p, p_p = parsing_hm36_gt_file(txtPath)

            # sample redundant video sequence
            img_index = np.arange(0, keypoints.shape[0], step=self.sample_step)
            for n_img_ in range(0, img_index.shape[0]):
                '''Fetch image and labels'''
                n_img = img_index[n_img_]
                assert keypoints.shape[1] == self.joint_num

                # project to image coordinate, and get 2D&3D label
                pt_2d = np.zeros((self.joint_num, 2), dtype=np.float)
                pt_3d = np.zeros((self.joint_num, 3), dtype=np.float)
                for n_jt in range(0, self.joint_num):
                    # tranform 3d joints from world space to camera space
                    pt_3d[n_jt] = np.dot(rot, keypoints[n_img, n_jt] - trans)       
                    # project 3d joints into image 2d joints
                    pt_2d[n_jt, 0], pt_2d[n_jt, 1] = CamProj(pt_3d[n_jt, 0], pt_3d[n_jt, 1], pt_3d[n_jt, 2], fl[0], fl[1], c_p[0], c_p[1])

                # substract root and remove
                pelvis3d = pt_3d[s_36_root_jt_idx]
                pt_3d = pt_3d - pelvis3d
                pt_3d = np.delete(pt_3d, 0, axis=0) #(16,3)

                # remove neck/nose
                pt_2d = np.delete(pt_2d, 9, axis=0) #(16,2)

                noise = np.random.normal(0,config.DATASET.sigma,32).reshape((16,2))
                pt_2d = pt_2d + noise
                
                gt_db.append({
                    'joints_2d': pt_2d, # [org_img_x, org_img_y]
                    'joints_3d': pt_3d, # [X, Y, Z] in camera coordinate
                    'folder': n_folder
                })

        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_db, fid, pickle.HIGHEST_PROTOCOL)
        logger.info('{} samples ared wrote {}'.format(len(gt_db), cache_file))
        return gt_db

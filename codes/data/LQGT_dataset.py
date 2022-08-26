import random
import numpy as np
import cv2
import lmdb
import torch
import torch.utils.data as data
import data.util as util
from data.data_loader import noiseDataset


class LQGTDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    # When training the code, it sent a dataset option. (train_bicubic_noise.yml)
    def __init__(self, opt):
        
        
        
        # 관련 인자들 초기화 
        super(LQGTDataset, self).__init__()
        self.opt = opt
        self.data_type = self.opt['data_type']
        self.paths_LQ, self.paths_GT = None, None
        self.sizes_LQ, self.sizes_GT = None, None
        self.LQ_env, self.GT_env = None, None  # environment for lmdb

        self.paths_GT, self.sizes_GT = util.get_image_paths(self.data_type, opt['dataroot_GT'])
        self.paths_LQ, self.sizes_LQ = util.get_image_paths(self.data_type, opt['dataroot_LQ'])
        
        
        
        
        
        # GT Path 는 무조건 존재해야 힘 
        assert self.paths_GT, 'Error: GT path is empty.'
        
        # Case 1: LQ / GT path 모두 존재한다면, 
        # LQ, GT 길이가 같아야 하므로, 해당 부분 확인 
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
            
            
        self.random_scale_list = [1]
        # print(opt['aug'])
        if self.opt['phase'] == 'train':
            if opt['aug'] and 'noise' in opt['aug']: 
                
                # 일반적인 training 의 경우, 이 부분에 속함. 
                self.noises = noiseDataset(opt['noise_data'], opt['GT_size']/opt['scale'])



    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(self.opt['dataroot_GT'], readonly=True, lock=False, readahead=False,
                                meminit=False)
        self.LQ_env = lmdb.open(self.opt['dataroot_LQ'], readonly=True, lock=False, readahead=False,
                                meminit=False)

    def __getitem__(self, index):
        if self.data_type == 'lmdb':
            if (self.GT_env is None) or (self.LQ_env is None):
                self._init_lmdb()
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        if self.data_type == 'lmdb':
            resolution = [int(s) for s in self.sizes_GT[index].split('_')]
        else:
            resolution = None
            
        #############################################
        #############################################
        ##     Another path of reading images.     ##
        
        img_GT = util.read_img(self.GT_env, GT_path, resolution)
        
        #############################################
        #############################################
        
        
        
        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_GT = util.modcrop(img_GT, scale)
            
            
            
        # change color space if necessary
        # If I pass through rhis code, 
        # Then my YUV grayscale image will 
        # be gone (cv2.COLOR_GRAY2BGR)
        # Therefore, I should not sent a code here
        '''
        if self.opt['color']:             # Channel        # Colour space     # tensor image
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]
        '''





        # get LQ image
        # LQ path 가 존재한다면, (일반적인 경우, 여기에 속한다. )
        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            if self.data_type == 'lmdb':
                resolution = [int(s) for s in self.sizes_LQ[index].split('_')]
            else:
                resolution = None
            img_LQ = util.read_img(self.LQ_env, LQ_path, resolution)




        # NO LQ image
        # LQ path 가 존재하지 않는 다면,
        else:  # down-sampling on-the-fly
            # randomly scale during training
            if self.opt['phase'] == 'train':
                random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, random_scale, scale, thres):
                    rlt = int(n * random_scale)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, random_scale, scale, GT_size)
                W_s = _mod(W_s, random_scale, scale, GT_size)
                img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                # force to 3 channels
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LQ = util.imresize_np(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)









        if self.opt['phase'] == 'train':

            # if the image size is too small
            # 당장은 여기에 속하는 이미지 없음. 일단은 무시하되, 
            # resize_np 등 channel 개수에 맞게 수정해야 함. 
            ############################################
            ############################################
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LQ = util.imresize_np(img_GT, 1 / scale, True)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)
            ############################################
            ############################################
                    

            H, W, C = img_LQ.shape
            LQ_size = GT_size // scale

            # randomly crop
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # augmentation - flip, rotate
            # 일반적인 경우, 실행 함. 
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])


        # change color space if necessary
        # If I pass through this code, 
        # Then my YUV grayscale image will 
        # be gone (cv2.COLOR_GRAY2BGR)
        # Therefore, I should not sent a code here
        ''' Never doing this 
        if self.opt['color']:
            img_LQ = util.channel_convert(C, self.opt['color'],
                                          [img_LQ])[0]  # TODO during val no definition
        '''

        # B(0) G(1) R(2) to R(2) G(1) B(0), HWC to CHW, numpy to tensor
        # my images are grayscale, therefore I don't need to patch it. 
        # SKIP below process
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        
        # H(0) W(1) C(2) -> into 2, 0, 1 : C(2) H(0) W(1)
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()
        
        

        if self.opt['phase'] == 'train':
            # add noise to LR during train
            if self.opt['aug'] and 'noise' in self.opt['aug']:
                
                # 모아둔 noise 들에서 random 하게 noise 를 선택한 뒤, LR image 에 적용시킨다. 
                noise = self.noises[np.random.randint(0, len(self.noises))]
                img_LQ = torch.clamp(img_LQ + noise, 0, 1)

        if LQ_path is None:
            LQ_path = GT_path
        return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)

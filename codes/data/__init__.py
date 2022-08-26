'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    phase = dataset_opt['phase']
    if phase == 'train':
        
        # 분산 작업(distribution)
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
            
        # 분산 작업(distribution) 을 하지 않는 경우 
        # number of worker / batch size / shuffle 여부 등을 정의한 뒤, 
        # torch.DataLoader 를 사용하여 반환함. 
        else:
            num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])
            batch_size = dataset_opt['batch_size']
            shuffle = True
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                           num_workers=num_workers, sampler=sampler, drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
                                           pin_memory=True)







######################################################
######################################################
# training 시 dataset 을 주어진 .png 파일로부터 만드는 방법 # 
######################################################
######################################################

'''
opt[datasets]
    in train phase, 
        name: train
        mode: LGQT
        aug: noise
        noise_data: ../datasets/… (Corrupted noise 가 저장된 경로 (kernel estimation 을 위함))
        dataroot_GT: HR pair 쌍
        dataroot_LQ: LR pair 쌍 
        use_shuffle: True
        n_workers: 1 (per gpu)
        batch_size: 2
        GT_size: 128
        use flip/rot: true
        Colour space: RGB
'''

def create_dataset(dataset_opt):
    
    
    # mode: LGQT
    mode = dataset_opt['mode']
    
    # data: data/ folder 
    if mode == 'LR':
        from data.LR_dataset import LRDataset as D
    elif mode == 'LQGT':
        from data.LQGT_dataset import LQGTDataset as D
    # elif mode == 'LQGTseg_bg':
    #     from data.LQGT_seg_bg_dataset import LQGTSeg_BG_Dataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    
    
    
    
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''
    # 1. 관련 인자들 초기화 (option 에 속하는 것)
    # 2. LQ / GT path 모두 존재한다면, 
    #    LQ, GT 길이가 같아야 하므로, 해당 부분 확인 
    # 3. 또한, Corrupted_noise 에 모은 noise 들을 self.noise 에 저장하고 초기생성 종료. 
    dataset = D(dataset_opt)
    
    
    
    

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset

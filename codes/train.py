import os
import math
import argparse
import random
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model


def init_dist(backend='nccl', **kwargs):
    ''' initialization for distributed training'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def main():
    
    
    '''
    [Step 1]: 전반적인 preprocessing 과정 
    
    Step 1-1. parameter Parser 를 통해 전달된 parameter 
              분리 및 실행 option 저장 
    '''
    
    
    
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    
    
    
    
    
    
    

    #### distributed training settings
    # Current default training parameter, 
    # I only send a parameter [opt], therefore I don't need to process
    # below code. 
    ############################################################
    ############################################################
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    ############################################################
    ############################################################
    
    
    
    '''
    Step 1-2. training 도중 연결 끊김 대처를 위한 
              resume_state 확인 과정 
    '''

    #### loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None






    '''
    Step 1-3: Rank (기본값: 0) 를 기준으로 logger 를 만듦?  
    '''

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)
    
    
    
    
    
    
    
    '''
    Step 1.4: Seed configuration for creating random number
    '''
    #### random seed
    seed = opt['train']['manual_seed'] # seed = 10
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0: # 일반적인 경우, 여기에 속한다. 
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)





    # 둘중에 하나 선택하도록 진행 
    torch.backends.cudnn.benckmark = True
    # torch.backends.cudnn.deterministic = True
    
    
    
    
    
    
    
    '''
    [Step 2]: Creating Dataset for training and validation 
    '''

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    train_loader = None
    val_loader = None
    for phase, dataset_opt in opt['datasets'].items():
        
        '''
        opt[datasets] << dataset_opt
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
        
        if phase == 'train': # train 시 아래의 코드를 따름.  
            # print('\n\n\n\n\n\n\n\n', dataset_opt)
            
            #######################################
            ###### How to make a train set? #######
            
            # Mode 에 따라서 dataset 을 dataset/X_dataset 
            # 에 생성자로 넘겨 초기화해둔다. 
            train_set = create_dataset(dataset_opt)
            
            #######################################
            #######################################
            
            
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank, dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
                
                
                
                
            # Train loader    
            # 분산 작업(distribution) 을 하지 않는 경우 
            # number of worker / batch size / shuffle 여부 등을 정의한 뒤, 
            # torch.DataLoader 를 사용하여 반환함. 
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            
            
            
            
   
            
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    #### create model
    model = create_model(opt)
    
    
    
    
    
    
    # 재시작의 경우라면, 
    #### resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0





    # 초기 실행의 경우라면, 
    #### training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=opt['train']['warmup_iter'])

            #### training
            ## SRGAN_model.py 
            '''
            def feed_data(self, data, need_GT=True):
                self.var_L = data['LQ'].to(self.device)  # LQ
                if need_GT:
                    self.var_H = data['GT'].to(self.device)  # GT
                    input_ref = data['ref'] if 'ref' in data else data['GT']
                    self.var_ref = input_ref.to(self.device)
            '''
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)

            # validation
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0 and val_loader is not None:
                avg_psnr = val_pix_err_f = val_pix_err_nf = val_mean_color_err = 0.0
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['LQ_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data(val_data)
                    model.test()

                    visuals = model.get_current_visuals()
                    sr_img = util.tensor2img(visuals['SR'])  # uint8
                    gt_img = util.tensor2img(visuals['GT'])  # uint8

                    # Save SR images for reference
                    save_img_path = os.path.join(img_dir,
                                                 '{:s}_{:d}.png'.format(img_name, current_step))
                    util.save_img(sr_img, save_img_path)

                    # calculate PSNR
                    crop_size = opt['scale']
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    avg_psnr += util.calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)


                avg_psnr = avg_psnr / idx
                val_pix_err_f /= idx
                val_pix_err_nf /= idx
                val_mean_color_err /= idx

                # log
                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}'.format(
                    epoch, current_step, avg_psnr))
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('val_pix_err_f', val_pix_err_f, current_step)
                    tb_logger.add_scalar('val_pix_err_nf', val_pix_err_nf, current_step)
                    tb_logger.add_scalar('val_mean_color_err', val_mean_color_err, current_step)

            #### save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()

from PIL import Image
import numpy as np
import os.path as osp
import glob
import os
import argparse
import yaml

parser = argparse.ArgumentParser(description='create a dataset')
parser.add_argument('--dataset', default='df2k', type=str, help='selecting different datasets')
parser.add_argument('--artifacts', default='', type=str, help='selecting different artifacts type')
parser.add_argument('--cleanup_factor', default=2, type=int, help='downscaling factor for image cleanup')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[4], help='super resolution upscale factor')
opt = parser.parse_args()

# define input and target directories
with open('./preprocess/paths.yml', 'r') as stream:
    PATHS = yaml.safe_load(stream)


def noise_patch(img, sp, max_var, min_mean):
    # my image is already mode 'L' 
    # img = rgb_img.convert('L')
    # rgb_img = img.convert('RGB')
    # rgb_img = np.array(rgb_img)
    
    img = np.array(img)

    w, h = img.shape
    collect_patchs = []

    for i in range(0, w - sp, sp):
        for j in range(0, h - sp, sp):
            patch = img[i:i + sp, j:j + sp]
            
            var_global = np.var(patch)
            mean_global = np.mean(patch)
            if var_global < max_var and mean_global > min_mean:
                # We don't need RGB patch, because we don't have any RGB image
                collect_patchs.append(patch)
                '''
                rgb_patch = rgb_img[i:i + sp, j:j + sp, :]
                collect_patchs.append(rgb_patch)
                '''
            

    return collect_patchs


if __name__ == '__main__':

    if opt.dataset == 'yuv':
        # img_dir: source which contain noise. 
        img_dir = PATHS[opt.dataset][opt.artifacts]['source']
        # noise_dir: noise which extracted from source img. 
        noise_dir = PATHS['datasets']['yuv'] + '/Corrupted_noise'
        
        ####################  
        ## HYPERPARAMETER ## 
        sp = 256
        max_var = 20
        min_mean = 0
        ####################
        ####################
        
    else:
        img_dir = PATHS[opt.dataset][opt.artifacts]['hr']['train']
        noise_dir = PATHS['datasets']['dped'] + '/DPEDiphone_noise'
        sp = 256
        max_var = 20
        min_mean = 50

    # There SHOULD NOT BE EXIST the noise path. 
    assert not os.path.exists(noise_dir)
    os.mkdir(noise_dir)

    img_paths = sorted(glob.glob(osp.join(img_dir, '*.png')))
    cnt = 0
    for path in img_paths:
        img_name = osp.splitext(osp.basename(path))[0]
        print('**********', img_name, '**********')
        
        # If you use YUV one channel image, 
        # It will automatically read as mode 'L' 
        img = Image.open(path)
        patchs = noise_patch(img, sp, max_var, min_mean)
        
        for idx, patch in enumerate(patchs):
            save_path = osp.join(noise_dir, '{}_{:03}.png'.format(img_name, idx))
            cnt += 1
            print('collect:', cnt, save_path)
            Image.fromarray(patch).save(save_path)

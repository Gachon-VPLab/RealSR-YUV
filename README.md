## RealSR - YUV 
RealSR-YUV is retrained [RealSR](https://github.com/Tencent/Real-SR) to feed YUV data directly into model. 

## Data preparation  
0. Move to directory  
`cd codes`

1. Generate HR-LR pair  
`!python3 ./preprocess/create_bicubic_dataset.py --dataset yuv --artifacts onec`

2. Extract noise data from source  
`!python3 ./preprocess/collect_noise.py --dataset yuv --artifact onec`  

3. Train  
`!CUDA_VISIBLE_DEVICES=0 python3 train.py -opt options/yuv/train_bicubic_noise.yml`  

4. Test  
`!CUDA_VISIBLE_DEVICES=0 python3 ./test_yuv_psnr.py -opt ./options/yuv/test_yuv_psnr.yml`

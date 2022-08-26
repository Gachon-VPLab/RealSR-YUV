import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']

    if model == 'sr':
        from .SR_model import SRModel as M
        
        
        
    # 일반적인 경우, 여기에 속한다. 
    elif model == 'srgan':
        from .SRGAN_model import SRGANModel as M
        
        
        
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    
    
    
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
